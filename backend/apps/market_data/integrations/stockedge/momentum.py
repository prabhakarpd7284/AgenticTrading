"""Momentum analytics over a StockEdge stock-wise score universe (e.g. Nifty 500).

Turns an ingested ``momentum_scores_*`` snapshot (``StockEdgeScanRow`` rows with
1M/3M/6M score + zone in ``attrs``) into an actionable, ranked momentum shortlist:

* **composite** — recency-weighted blend of the 1M/3M/6M scores (0-100).
* **acceleration** — 1M score minus 3M score; >0 means short-term momentum is
  pulling ahead of the medium term (improving), <0 means it's rolling over.
* **classification** — a trader-readable state from the three zones:
    - ``sustained`` : Bullish on all three horizons (strong, durable trend)
    - ``emerging``  : Bullish 1M but not yet 6M (fresh momentum turning up)
    - ``fading``    : 1M no longer Bullish while 3M/6M were (momentum rolling over)
    - ``neutral``   : everything else

Pure read layer: no writes, no Django model changes. Returns plain dicts so the
same output feeds a management command, a DRF endpoint, or the screener.
"""
from __future__ import annotations

from typing import Any, Iterable

# Recency-weighted blend — the near term carries the most weight for intraday/
# swing decisions, the longer horizons confirm durability.
COMPOSITE_WEIGHTS = {"1m": 0.5, "3m": 0.3, "6m": 0.2}
HORIZONS = ("1m", "3m", "6m")


def latest_momentum_snapshot(index_slug: str = "nifty_500"):
    """Most recent ``momentum_scores_<index>`` snapshot, or None."""
    from apps.market_data.models import StockEdgeSnapshot

    return (
        StockEdgeSnapshot.objects
        .filter(dataset=f"momentum_scores_{index_slug}")
        .order_by("-as_of_date")
        .first()
    )


def _score(attrs: dict, horizon: str) -> float | None:
    v = attrs.get(f"{horizon}_score")
    return v if isinstance(v, (int, float)) else None


def _zone(attrs: dict, horizon: str) -> str | None:
    return attrs.get(f"{horizon}_score_zone")


def composite_score(attrs: dict) -> float | None:
    """Recency-weighted blend of available horizon scores, renormalized."""
    num = 0.0
    wsum = 0.0
    for h, w in COMPOSITE_WEIGHTS.items():
        s = _score(attrs, h)
        if s is not None:
            num += w * s
            wsum += w
    return round(num / wsum, 1) if wsum else None


def acceleration(attrs: dict) -> float | None:
    """1M minus 3M score: positive = accelerating, negative = decelerating."""
    s1, s3 = _score(attrs, "1m"), _score(attrs, "3m")
    if s1 is None or s3 is None:
        return None
    return round(s1 - s3, 1)


def classify(attrs: dict) -> str:
    z1, z3, z6 = _zone(attrs, "1m"), _zone(attrs, "3m"), _zone(attrs, "6m")
    bull = "Bullish"
    if z1 == bull and z3 == bull and z6 == bull:
        return "sustained"
    if z1 == bull and z6 != bull:
        return "emerging"
    if z1 != bull and (z3 == bull or z6 == bull):
        return "fading"
    return "neutral"


def enrich(row) -> dict:
    """Project one ``StockEdgeScanRow`` into a momentum record (plain dict)."""
    a = row.attrs or {}
    return {
        "symbol": row.symbol,
        "name": row.name,
        "sector": row.sector,
        "industry": row.industry,
        "ltp": row.ltp,
        "change_pct": row.change_pct,
        "market_cap_cr": row.market_cap_cr,
        "score_1m": _score(a, "1m"), "zone_1m": _zone(a, "1m"),
        "score_3m": _score(a, "3m"), "zone_3m": _zone(a, "3m"),
        "score_6m": _score(a, "6m"), "zone_6m": _zone(a, "6m"),
        "composite": composite_score(a),
        "acceleration": acceleration(a),
        "classification": classify(a),
    }


def momentum_universe(snapshot) -> list[dict]:
    """All enriched rows for a snapshot, ranked by composite score (desc)."""
    rows = [enrich(r) for r in snapshot.scan_rows.all()]
    rows.sort(key=lambda r: (r["composite"] is not None, r["composite"] or 0), reverse=True)
    return rows


# ── filters / shortlists (compose freely) ─────────────────────────────
def filter_universe(
    rows: Iterable[dict],
    *,
    classification: str | None = None,
    min_composite: float | None = None,
    min_acceleration: float | None = None,
    min_mcap_cr: float | None = None,
    sector: str | None = None,
) -> list[dict]:
    out = []
    for r in rows:
        if classification and r["classification"] != classification:
            continue
        if min_composite is not None and (r["composite"] or 0) < min_composite:
            continue
        if min_acceleration is not None and (r["acceleration"] is None
                                             or r["acceleration"] < min_acceleration):
            continue
        if min_mcap_cr is not None and (r["market_cap_cr"] or 0) < min_mcap_cr:
            continue
        if sector and (r["sector"] or "").lower() != sector.lower():
            continue
        out.append(r)
    return out


def shortlist(rows: list[dict], *, top: int = 15, **filters: Any) -> list[dict]:
    """Top-N momentum names after applying filters (rows already ranked)."""
    return filter_universe(rows, **filters)[:top]


def universe_summary(rows: list[dict]) -> dict:
    """Counts + breadth for the whole universe (for headers/overlay)."""
    def zone_count(horizon: str, zone: str) -> int:
        return sum(1 for r in rows if r[f"zone_{horizon}"] == zone)

    classes: dict[str, int] = {}
    for r in rows:
        classes[r["classification"]] = classes.get(r["classification"], 0) + 1

    n = len(rows) or 1
    return {
        "count": len(rows),
        "classes": classes,
        "bullish_1m": zone_count("1m", "Bullish"),
        "bullish_1m_pct": round(100 * zone_count("1m", "Bullish") / n, 1),
        "avg_composite": round(
            sum(r["composite"] or 0 for r in rows) / n, 1
        ),
    }
