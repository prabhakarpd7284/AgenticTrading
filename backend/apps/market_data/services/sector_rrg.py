"""Sector Rotation RRG (Relative Rotation Graph) vs NIFTY.

For each NIFTY sector index, compute:

  rs_ratio     14-day relative strength vs NIFTY 50, normalised to 100
  rs_momentum  rate-of-change of rs_ratio over the last 5 days
  tail         last 10 weekly (rs_ratio, rs_momentum) points so the UI can
               draw the trail
  quadrant     LEADING / WEAKENING / LAGGING / IMPROVING

Quadrants follow the Bloomberg RRG convention:
    rs_ratio ≥ 100 AND rs_mom ≥ 0  → LEADING
    rs_ratio ≥ 100 AND rs_mom <  0 → WEAKENING
    rs_ratio <  100 AND rs_mom <  0 → LAGGING
    rs_ratio <  100 AND rs_mom ≥ 0  → IMPROVING
"""
from __future__ import annotations

import statistics
from typing import Any

from django.core.cache import cache

_TTL = 300
_LOOKBACK_DAYS = 60      # need ~50 daily closes to compute 10 weekly RRG points


def _yf_history(yf_symbol: str, days: int = _LOOKBACK_DAYS) -> list[float]:
    """Return last N daily closes for a yfinance ticker."""
    key = f"rrg:closes:{yf_symbol}:{days}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        import yfinance as yf
        ticker = yf.Ticker(yf_symbol)
        hist = ticker.history(period=f"{days + 10}d", auto_adjust=False)
        closes = [float(c) for c in hist["Close"].dropna().tolist()]
        out = closes[-days:]
        cache.set(key, out, _TTL)
        return out
    except Exception:  # noqa: BLE001
        cache.set(key, [], _TTL); return []


def _rs_ratio_series(sector_closes: list[float], bench_closes: list[float]) -> list[float]:
    """Normalised ratio (sector / bench), rebased so the FIRST value = 100."""
    n = min(len(sector_closes), len(bench_closes))
    if n == 0:
        return []
    s, b = sector_closes[-n:], bench_closes[-n:]
    raw = [s[i] / b[i] for i in range(n) if b[i] > 0]
    if not raw:
        return []
    base = raw[0]
    return [round(r / base * 100.0, 3) for r in raw] if base > 0 else []


def _quadrant(rs_ratio: float, rs_mom: float) -> str:
    if rs_ratio >= 100 and rs_mom >= 0:   return "LEADING"
    if rs_ratio >= 100 and rs_mom <  0:   return "WEAKENING"
    if rs_ratio <  100 and rs_mom <  0:   return "LAGGING"
    return "IMPROVING"


def build_sector_rrg(tenant=None) -> dict[str, Any]:
    from apps.market_data.services.pulse_service import SECTOR_TICKERS

    bench = _yf_history("^NSEI")    # NIFTY 50
    if not bench:
        return {"count": 0, "rows": [], "note": "NIFTY benchmark data unavailable."}

    rows: list[dict] = []
    for sector_key, yf_sym in SECTOR_TICKERS.items():
        closes = _yf_history(yf_sym)
        if not closes:
            rows.append({"sector": sector_key, "quadrant": "no_data",
                         "rs_ratio": 0.0, "rs_momentum": 0.0, "tail": []})
            continue

        ratio_series = _rs_ratio_series(closes, bench)
        if len(ratio_series) < 11:
            rows.append({"sector": sector_key, "quadrant": "no_data",
                         "rs_ratio": 0.0, "rs_momentum": 0.0, "tail": []})
            continue

        # Approximate weekly samples by taking every 5th element from the tail.
        weekly = ratio_series[::5][-10:]
        # Momentum = current rs_ratio − 5-day-ago rs_ratio
        rs_now = ratio_series[-1]
        rs_5ago = ratio_series[-6] if len(ratio_series) >= 6 else ratio_series[0]
        rs_mom = round(rs_now - rs_5ago, 3)

        # Build (rs_ratio, rs_mom) tail
        tail = []
        for i, val in enumerate(weekly):
            offset = max(0, len(ratio_series) - (len(weekly) - i) * 5 - 1)
            past_5ago = ratio_series[max(0, offset - 5)]
            tail.append({"rs_ratio": val, "rs_mom": round(val - past_5ago, 3)})

        rows.append({
            "sector": sector_key,
            "yf_symbol": yf_sym,
            "rs_ratio": rs_now,
            "rs_momentum": rs_mom,
            "quadrant": _quadrant(rs_now, rs_mom),
            "tail": tail,
        })

    quadrant_order = {"LEADING": 0, "IMPROVING": 1, "WEAKENING": 2, "LAGGING": 3, "no_data": 4}
    rows.sort(key=lambda r: (quadrant_order.get(r["quadrant"], 4), -r.get("rs_ratio", 0)))
    return {
        "count": len(rows),
        "rows": rows,
        "note": (
            "LEADING (top-right) sectors are outperforming and accelerating — "
            "trade their constituents long. IMPROVING (bottom-right) are early "
            "rotation candidates. LAGGING/WEAKENING = avoid or short the "
            "weakest names."
        ),
    }
