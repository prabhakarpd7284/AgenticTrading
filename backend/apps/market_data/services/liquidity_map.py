"""Liquidity & slippage cost map.

For every symbol currently held or watchlisted, surface:
  - bid, ask, spread_bps        (via legacy BrokerClient ltp_data when supported)
  - depth_imbalance in [-1, 1]   (placeholder when full depth isn't exposed)
  - avg_historical_slippage_bps (derived on the fly from TradeJournal
                                 entry_price vs fill_price)

Failures degrade gracefully — a symbol with no live quote returns 0s
instead of vanishing so the UI can still surface it as "no data".

Performance:
  - Per-symbol quote is cached in Django cache for 60s so a tab reload
    doesn't burn N broker calls.
  - Symbol set is capped (default 20) so the page resolves in <2s.
"""
from __future__ import annotations

import statistics
from typing import Any

from django.core.cache import cache

_QUOTE_TTL = 60
_MAX_SYMBOLS = 20


def _empty_quote() -> dict[str, float]:
    return {"bid": 0.0, "ask": 0.0}


def _quote_from_ltp(ltp: float) -> dict[str, float]:
    if ltp <= 0:
        return _empty_quote()
    spread = max(ltp * 0.0005, 0.05)
    return {"bid": round(ltp - spread / 2, 2), "ask": round(ltp + spread / 2, 2)}


def _batch_live_quotes(symbols: list[str]) -> dict[str, dict[str, float]]:
    """One broker round-trip for up to 50 symbols. Cached individually for 60s."""
    out: dict[str, dict[str, float]] = {}
    miss: list[str] = []
    for sym in symbols:
        v = cache.get(f"liq:quote:{sym}")
        if v is not None:
            out[sym] = v
        else:
            miss.append(sym)

    if miss:
        try:
            from trading.services.data_service import BrokerClient
            broker = BrokerClient.get_instance(); broker.ensure_login()
            rows = broker.fetch_batch_ltp(miss) or []
            got: dict[str, float] = {r["symbol"]: float(r.get("ltp", 0) or 0) for r in rows}
            for sym in miss:
                q = _quote_from_ltp(got.get(sym, 0.0))
                out[sym] = q
                cache.set(f"liq:quote:{sym}", q, _QUOTE_TTL)
        except Exception:  # noqa: BLE001
            for sym in miss:
                out[sym] = _empty_quote()
                cache.set(f"liq:quote:{sym}", out[sym], _QUOTE_TTL)
    return out


def _historical_slippage_bps(symbol: str) -> float:
    """Derive slippage from TradeJournal entry_price vs fill_price.

    TradeJournal doesn't store slippage_bps as a column, so we compute it on the fly.
    Returns the rolling 50-fill mean |slippage| in bps.
    """
    from trading.models import TradeJournal
    try:
        rows = TradeJournal.objects.filter(symbol=symbol).exclude(
            fill_price__isnull=True
        ).values_list("entry_price", "fill_price")[:50]
        bps: list[float] = []
        for entry, fill in rows:
            if entry and fill and entry > 0:
                bps.append(abs((float(fill) - float(entry)) / float(entry)) * 10_000.0)
        if not bps:
            return 0.0
        return round(statistics.mean(bps), 2)
    except Exception:  # noqa: BLE001
        return 0.0


# Standard intraday buckets used by the time-of-day heatmap.
_TOD_BUCKETS = [
    ("open",    9, 15, 10, 0),
    ("morning", 10, 0, 12, 0),
    ("midday",  12, 0, 13, 30),
    ("afternoon", 13, 30, 14, 45),
    ("close",   14, 45, 15, 30),
]


def _slippage_by_time_of_day(symbol: str) -> dict[str, float]:
    """Mean |slippage| in bps bucketed by intraday time-of-day."""
    from trading.models import TradeJournal
    out: dict[str, list[float]] = {b[0]: [] for b in _TOD_BUCKETS}
    try:
        rows = TradeJournal.objects.filter(symbol=symbol).exclude(
            fill_price__isnull=True
        ).values_list("entry_price", "fill_price", "created_at")[:200]
        for entry, fill, ts in rows:
            if not (entry and fill and ts and entry > 0):
                continue
            bps = abs((float(fill) - float(entry)) / float(entry)) * 10_000.0
            t = ts.timetz()
            for name, h1, m1, h2, m2 in _TOD_BUCKETS:
                if (h1, m1) <= (t.hour, t.minute) < (h2, m2):
                    out[name].append(bps); break
        return {
            name: round(statistics.mean(vals), 2) if vals else 0.0
            for name, vals in out.items()
        }
    except Exception:  # noqa: BLE001
        return {name: 0.0 for name, *_ in _TOD_BUCKETS}


def _collect_symbols(tenant=None) -> list[str]:
    symbols: set[str] = set()
    try:
        from trading.models import TradeJournal, WatchlistEntry, StraddlePosition
        for s in TradeJournal.objects.filter(status__in=("EXECUTED", "PAPER", "FILLED")).values_list("symbol", flat=True)[:200]:
            if s:
                symbols.add(s)
        for s in WatchlistEntry.objects.values_list("symbol", flat=True)[:200]:
            if s:
                symbols.add(s)
        for p in StraddlePosition.objects.filter(status="ACTIVE"):
            if p.ce_symbol: symbols.add(p.ce_symbol)
            if p.pe_symbol: symbols.add(p.pe_symbol)
    except Exception:  # noqa: BLE001
        pass
    return sorted(symbols)


def build_liquidity_map(tenant=None) -> dict[str, Any]:
    symbols = _collect_symbols(tenant)[:_MAX_SYMBOLS]
    quotes = _batch_live_quotes(symbols)
    rows: list[dict] = []
    for sym in symbols:
        q = quotes.get(sym, _empty_quote())
        bid, ask = q["bid"], q["ask"]
        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0.0
        spread_bps = ((ask - bid) / mid * 10_000) if mid > 0 else 0.0
        rows.append({
            "symbol": sym,
            "bid": bid,
            "ask": ask,
            "mid": round(mid, 2),
            "spread_bps": round(spread_bps, 2),
            "depth_imbalance": 0.0,
            "avg_historical_slippage_bps": _historical_slippage_bps(sym),
            "slippage_by_tod": _slippage_by_time_of_day(sym),
        })
    return {
        "count": len(rows),
        "rows": rows,
        "note": (
            f"Showing top {len(rows)} of {len(_collect_symbols(tenant))} subscribed symbols. "
            "Bid/ask is approximated from LTP ± 5 bps (broker SDK doesn't expose "
            "level-2 depth in this build). Slippage is the rolling 50-fill mean "
            "derived from TradeJournal entry_price vs fill_price."
        ),
    }
