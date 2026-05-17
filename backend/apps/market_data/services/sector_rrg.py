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
_WEEKLY_LOOKBACK_DAYS = 130     # ~26 weeks → enough for 13-week tails


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


def build_sector_rrg(tenant=None, *, weekly: bool = False) -> dict[str, Any]:
    """When `weekly=True`, samples every 5th daily close (≈ weekly) and uses
    13 points of tail; otherwise daily resolution with 10-point tail."""
    from apps.market_data.services.pulse_service import SECTOR_TICKERS

    lookback = _WEEKLY_LOOKBACK_DAYS if weekly else _LOOKBACK_DAYS
    tail_len = 13 if weekly else 10
    momentum_lookback = 25 if weekly else 5  # 5w vs 5d

    bench = _yf_history("^NSEI", days=lookback)
    if not bench:
        return {"count": 0, "rows": [], "mode": "weekly" if weekly else "daily",
                "note": "NIFTY benchmark data unavailable."}

    rows: list[dict] = []
    for sector_key, yf_sym in SECTOR_TICKERS.items():
        closes = _yf_history(yf_sym, days=lookback)
        if not closes:
            rows.append({"sector": sector_key, "quadrant": "no_data",
                         "rs_ratio": 0.0, "rs_momentum": 0.0, "tail": []})
            continue

        ratio_series = _rs_ratio_series(closes, bench)
        if len(ratio_series) < momentum_lookback + 1:
            rows.append({"sector": sector_key, "quadrant": "no_data",
                         "rs_ratio": 0.0, "rs_momentum": 0.0, "tail": []})
            continue

        # Sample every 5th element if weekly, otherwise keep daily.
        step = 5 if weekly else 1
        sampled = ratio_series[::step][-tail_len:]
        rs_now = ratio_series[-1]
        rs_back = ratio_series[-(momentum_lookback + 1)]
        rs_mom = round(rs_now - rs_back, 3)

        tail = []
        for i, val in enumerate(sampled):
            offset = max(0, len(ratio_series) - (len(sampled) - i) * step - 1)
            past = ratio_series[max(0, offset - momentum_lookback)]
            tail.append({"rs_ratio": val, "rs_mom": round(val - past, 3)})

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
        "mode": "weekly" if weekly else "daily",
        "tail_length": tail_len,
        "rows": rows,
        "note": (
            "LEADING (top-right) sectors are outperforming and accelerating — "
            "trade their constituents long. IMPROVING (bottom-right) are early "
            "rotation candidates. LAGGING/WEAKENING = avoid or short the "
            "weakest names. " + ("Weekly mode with 13-week tail." if weekly else "Daily mode with 10-day tail.")
        ),
    }
