"""Liquidity & slippage cost map.

For every symbol currently held or watchlisted, surface:
  - bid, ask, spread_bps        (via legacy BrokerClient ltp_data when supported)
  - depth_imbalance in [-1, 1]   (placeholder when full depth isn't exposed)
  - avg_historical_slippage_bps (from TradeJournal slippage_bps when present)

Failures degrade gracefully — a symbol with no live quote returns 0s
instead of vanishing so the UI can still surface it as "no data".
"""
from __future__ import annotations

import statistics
from typing import Any


def _live_quote(symbol: str) -> dict[str, float]:
    """Best-effort quote pull via the legacy stack."""
    try:
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service

        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(symbol)
        if not token:
            return {"bid": 0.0, "ask": 0.0}
        r = broker.ltp("NSE", symbol, token)
        if not isinstance(r, dict):
            return {"bid": 0.0, "ask": 0.0}
        ltp = float(r.get("ltp", 0) or 0)
        # Without full depth, approximate bid/ask as ltp ± 0.05% (10bps) tick.
        if ltp <= 0:
            return {"bid": 0.0, "ask": 0.0}
        spread = max(ltp * 0.0005, 0.05)
        return {"bid": round(ltp - spread / 2, 2), "ask": round(ltp + spread / 2, 2)}
    except Exception:  # noqa: BLE001
        return {"bid": 0.0, "ask": 0.0}


def _historical_slippage_bps(symbol: str) -> float:
    from trading.models import TradeJournal
    try:
        rows = list(TradeJournal.objects.filter(symbol=symbol).values_list("slippage_bps", flat=True))
        rows = [abs(float(r)) for r in rows if r is not None]
        if not rows:
            return 0.0
        return round(statistics.mean(rows[-50:]), 2)
    except Exception:  # noqa: BLE001
        return 0.0


def _collect_symbols(tenant=None) -> list[str]:
    symbols: set[str] = set()
    try:
        from trading.models import TradeJournal, WatchlistEntry, StraddlePosition
        for s in TradeJournal.objects.filter(status__in=("EXECUTED", "PAPER")).values_list("symbol", flat=True)[:200]:
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
    symbols = _collect_symbols(tenant)
    rows: list[dict] = []
    for sym in symbols[:50]:    # cap broker calls — front-end shows top 50
        q = _live_quote(sym)
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
        })
    return {
        "count": len(rows),
        "rows": rows,
        "note": "Bid/ask is approximated from LTP ± 5 bps because the broker SDK doesn't expose level-2 depth in this build. Slippage is the rolling 50-fill mean from TradeJournal.",
    }
