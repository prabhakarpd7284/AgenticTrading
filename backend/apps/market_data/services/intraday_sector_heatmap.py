"""Intraday Sector Dispersion & Capital-Rotation Heatmap (5-min buckets).

For each 5-min slot between 09:15 and 15:30 IST, compute per-sector:

  dispersion    cross-sectional stddev of constituent % moves
  median_pct    median % move of constituents
  leader_pct    top-mover % within sector
  laggard_pct   bottom-mover % within sector

Builds a sector × slot matrix so the FE can render a heatmap (hot cells
= dispersion-rich slots where stock-picking is paying right now).

Heavy on broker calls when symbols × bars × sectors all expand — this
service is deliberately capped to top-2-constituents-per-sector to stay
under the SDK's rate limits in paper mode.
"""
from __future__ import annotations

import statistics
from datetime import date, time
from typing import Any

from django.core.cache import cache

from apps.market_data.services.orb_tracker import _parse_minute


_TTL = 300
_SLOT_MIN = 5
_SLOT_END = time(15, 30)


def _bucket_label(t: time) -> str:
    minute = (t.minute // _SLOT_MIN) * _SLOT_MIN
    return f"{t.hour:02d}:{minute:02d}"


def _fetch_5m(symbol: str) -> list[dict]:
    key = f"sector_heatmap:5m:{symbol}:{date.today().isoformat()}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service
        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(symbol)
        if not token:
            cache.set(key, [], _TTL); return []
        today = date.today()
        start = today.strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 15:30")
        try:
            raw = broker.fetch_candles(token, start, end, "FIVE_MINUTE", exchange="NSE") or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "FIVE_MINUTE") or []
        out = [
            {"t": str(r[0]), "o": float(r[1]), "c": float(r[4])}
            for r in raw if len(r) >= 5
        ]
        cache.set(key, out, _TTL); return out
    except Exception:  # noqa: BLE001
        cache.set(key, [], _TTL); return []


def _pct_per_slot(bars: list[dict]) -> dict[str, float]:
    """Bar-by-bar % change keyed by 5-min slot string."""
    out: dict[str, float] = {}
    for b in bars:
        t = _parse_minute(b["t"])
        if t is None or t >= _SLOT_END:
            continue
        if b["o"] > 0:
            out[_bucket_label(t)] = round((b["c"] - b["o"]) / b["o"] * 100.0, 3)
    return out


def build_intraday_sector_heatmap(tenant=None) -> dict[str, Any]:
    from apps.market_data.services.rotation_service import SECTOR_CONSTITUENTS

    # Cap to 2 names per sector to keep broker calls reasonable.
    sector_syms: dict[str, list[str]] = {
        sector: list(syms)[:2] for sector, syms in SECTOR_CONSTITUENTS.items()
    }
    all_syms = sorted({s for v in sector_syms.values() for s in v})

    pct_by_sym: dict[str, dict[str, float]] = {}
    for sym in all_syms:
        pct_by_sym[sym] = _pct_per_slot(_fetch_5m(sym))

    # Collect every slot label any symbol produced (already chronological by
    # virtue of HH:MM string sort).
    all_slots: set[str] = set()
    for s in pct_by_sym.values():
        all_slots.update(s.keys())
    ordered_slots = sorted(all_slots)

    rows: list[dict] = []
    for sector, syms in sector_syms.items():
        cells: list[dict] = []
        for slot in ordered_slots:
            values = [pct_by_sym[s].get(slot) for s in syms if pct_by_sym[s].get(slot) is not None]
            if not values:
                cells.append({"slot": slot, "n": 0, "median": 0.0,
                              "dispersion": 0.0, "leader": 0.0, "laggard": 0.0})
                continue
            cells.append({
                "slot": slot,
                "n": len(values),
                "median": round(statistics.median(values), 3),
                "dispersion": round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
                "leader": round(max(values), 3),
                "laggard": round(min(values), 3),
            })
        rows.append({"sector": sector, "cells": cells})

    return {
        "slots": ordered_slots,
        "rows": rows,
        "sector_count": len(rows),
        "slot_count": len(ordered_slots),
        "note": (
            "Per 5-min bucket: dispersion = cross-stock std-dev within the "
            "sector. Hot cells (>1.5) = stock-picking pays *right now* in "
            "that sector. Run a long/short within the sector on those slots."
        ),
    }
