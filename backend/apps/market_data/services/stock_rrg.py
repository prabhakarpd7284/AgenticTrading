"""Stock-level RRG — individual stocks vs NIFTY 50.

Same math as sector_rrg but the cohort is per-stock instead of per-sector.
By default scores: open positions + watchlist symbols (capped at 25). The
caller can pass `?symbols=A,B,C` to override.

Always returns weekly mode with 8-week tails — the right horizon for a
swing trader picking which name to add or trim.
"""
from __future__ import annotations

from typing import Any

from apps.market_data.services.sector_rrg import (
    _yf_history, _rs_ratio_series, _quadrant,
)

_LOOKBACK_DAYS = 130     # ~26 weeks → enough for 8-week tails + 5w momentum
_TAIL = 8
_MOMENTUM_LOOKBACK = 25  # weeks expressed in trading days


def _to_yf(symbol: str) -> str:
    """NSE equity → yfinance ticker (Reliance → RELIANCE.NS)."""
    s = symbol.strip().upper()
    if s.endswith(".NS") or s.startswith("^"):
        return s
    return f"{s}.NS"


def _default_cohort() -> list[str]:
    syms: set[str] = set()
    try:
        from trading.models import WatchlistEntry, TradeJournal
        for s in WatchlistEntry.objects.values_list("symbol", flat=True)[:40]:
            if s: syms.add(s.upper())
        for s in TradeJournal.objects.filter(
            status__in=("EXECUTED", "PAPER", "APPROVED")
        ).values_list("symbol", flat=True)[:40]:
            if s: syms.add(s.upper())
    except Exception:  # noqa: BLE001
        pass
    return sorted(syms)


def build_stock_rrg(symbols: list[str] | None = None) -> dict[str, Any]:
    cohort = (symbols or _default_cohort())[:25]
    bench = _yf_history("^NSEI", days=_LOOKBACK_DAYS)
    if not bench:
        return {"count": 0, "rows": [],
                "note": "NIFTY benchmark data unavailable (yfinance offline?)."}

    rows: list[dict] = []
    for sym in cohort:
        closes = _yf_history(_to_yf(sym), days=_LOOKBACK_DAYS)
        if not closes:
            rows.append({"symbol": sym, "quadrant": "no_data",
                         "rs_ratio": 0.0, "rs_momentum": 0.0, "tail": []})
            continue

        ratio_series = _rs_ratio_series(closes, bench)
        if len(ratio_series) < _MOMENTUM_LOOKBACK + 1:
            rows.append({"symbol": sym, "quadrant": "no_data",
                         "rs_ratio": 0.0, "rs_momentum": 0.0, "tail": []})
            continue

        # Weekly samples
        sampled = ratio_series[::5][-_TAIL:]
        rs_now = ratio_series[-1]
        rs_back = ratio_series[-(_MOMENTUM_LOOKBACK + 1)]
        rs_mom = round(rs_now - rs_back, 3)

        tail = []
        for i, val in enumerate(sampled):
            offset = max(0, len(ratio_series) - (len(sampled) - i) * 5 - 1)
            past = ratio_series[max(0, offset - _MOMENTUM_LOOKBACK)]
            tail.append({"rs_ratio": val, "rs_mom": round(val - past, 3)})

        rows.append({
            "symbol": sym,
            "rs_ratio": rs_now,
            "rs_momentum": rs_mom,
            "quadrant": _quadrant(rs_now, rs_mom),
            "tail": tail,
        })

    order = {"LEADING": 0, "IMPROVING": 1, "WEAKENING": 2, "LAGGING": 3, "no_data": 4}
    rows.sort(key=lambda r: (order.get(r["quadrant"], 4), -r.get("rs_ratio", 0)))
    return {
        "count": len(rows),
        "mode": "weekly",
        "tail_length": _TAIL,
        "rows": rows,
        "note": (
            "Per-stock RRG vs NIFTY 50 (weekly, 8-week tails). LEADING = "
            "outperforming and accelerating — add or hold. IMPROVING = "
            "starting to outperform — early-rotation candidate. WEAKENING "
            "= losing momentum — trim. LAGGING = avoid or short."
        ),
    }
