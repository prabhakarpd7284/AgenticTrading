"""Sector Dispersion & Intraday Leader/Laggard heatmap.

For each NIFTY sector, pull today's % change of every constituent and
compute:

  dispersion_pct    cross-sectional stddev of constituent returns
  leaders           top 3 by intraday %
  laggards          bottom 3 by intraday %
  median_pct        median return across constituents
  cohort_size       how many constituents we got data for

High dispersion means stock-picking is rewarded over sector beta.
"""
from __future__ import annotations

import statistics
from typing import Any

from django.core.cache import cache

_TTL = 60


def _live_pct_change(symbol: str) -> float:
    """Today's % change for a single equity symbol. Uses the same batch LTP
    helper liquidity_map already employs."""
    try:
        from trading.services.data_service import BrokerClient
        broker = BrokerClient.get_instance(); broker.ensure_login()
        rows = broker.fetch_batch_ltp([symbol]) or []
        if not rows:
            return 0.0
        return float(rows[0].get("pct_change", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0.0


def _batch_changes(symbols: list[str]) -> dict[str, float]:
    """One round-trip; cached per-symbol for 60s."""
    out: dict[str, float] = {}
    miss: list[str] = []
    for s in symbols:
        v = cache.get(f"disp:pct:{s}")
        if v is not None:
            out[s] = v
        else:
            miss.append(s)

    if miss:
        try:
            from trading.services.data_service import BrokerClient
            broker = BrokerClient.get_instance(); broker.ensure_login()
            rows = broker.fetch_batch_ltp(miss) or []
            got = {r["symbol"]: float(r.get("pct_change", 0) or 0) for r in rows}
            for s in miss:
                v = got.get(s, 0.0)
                out[s] = v
                cache.set(f"disp:pct:{s}", v, _TTL)
        except Exception:  # noqa: BLE001
            for s in miss:
                out[s] = 0.0
                cache.set(f"disp:pct:{s}", 0.0, _TTL)
    return out


def build_sector_dispersion(tenant=None) -> dict[str, Any]:
    from apps.market_data.services.rotation_service import SECTOR_CONSTITUENTS

    # Flatten + dedupe symbols so we issue one batch request
    all_syms: list[str] = []
    seen: set[str] = set()
    for syms in SECTOR_CONSTITUENTS.values():
        for s in syms:
            if s not in seen:
                seen.add(s); all_syms.append(s)

    pct = _batch_changes(all_syms)

    rows: list[dict] = []
    for sector, syms in SECTOR_CONSTITUENTS.items():
        cohort = [(s, pct.get(s, 0.0)) for s in syms if s in pct]
        if not cohort:
            rows.append({"sector": sector, "cohort_size": 0,
                         "median_pct": 0.0, "dispersion_pct": 0.0,
                         "leaders": [], "laggards": []})
            continue
        values = [v for _, v in cohort]
        med = round(statistics.median(values), 2) if values else 0.0
        disp = round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0
        cohort_sorted = sorted(cohort, key=lambda kv: kv[1], reverse=True)
        leaders = [{"symbol": s, "pct": round(v, 2)} for s, v in cohort_sorted[:3]]
        laggards = [{"symbol": s, "pct": round(v, 2)} for s, v in cohort_sorted[-3:][::-1]]
        rows.append({
            "sector": sector,
            "cohort_size": len(cohort),
            "median_pct": med,
            "dispersion_pct": disp,
            "leaders": leaders,
            "laggards": laggards,
        })

    # Highest dispersion first — that's where stock-picking pays best today.
    rows.sort(key=lambda r: -r["dispersion_pct"])
    return {
        "count": len(rows),
        "rows": rows,
        "note": (
            "Dispersion > 1.5% = stock-picking pays today; trade the leader "
            "and short the laggard within the same sector. Dispersion < 0.5% "
            "= sector beta is dominant; trade the sector ETF or skip."
        ),
    }
