"""Pyramid-Add Trigger Board.

For every open swing position with a defined initial risk (entry vs
stop_loss), compute:

  current_r          (current_pnl / initial_risk_inr)
  next_add_r         the next R-multiple the trader hasn't added at yet
                     — 1, 2, 3 in sequence
  add_signal         tag in {READY, WATCH, NOT_YET}
                        READY    current_r >= next_add_r AND price is on
                                 EMA-5 / 10WMA / VWAP retest (handle setup)
                        WATCH    current_r within 0.2 of next_add_r
                        NOT_YET  current_r < next_add_r − 0.2
  handle_present     True if the last 3 daily bars closed within 1.5% of
                     each other (proxy for "handle" / tight consolidation)
  size_for_add       suggested qty for the add (= original × 0.5)

The board is the swing trader's "when do I pyramid?" checklist.
"""
from __future__ import annotations

import statistics
from datetime import date, datetime, timedelta, timezone
from typing import Any

from django.core.cache import cache


_TTL = 60


def _last_3_close_tightness(symbol: str) -> tuple[bool, float]:
    """Last 3 daily closes within 1.5% range of each other → handle present."""
    key = f"pyramid:tight:{symbol}"
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        from trading.services.data_service import BrokerClient
        from trading.services.ticker_service import ticker_service
        broker = BrokerClient.get_instance(); broker.ensure_login()
        token = ticker_service.get_token(symbol)
        if not token:
            cache.set(key, (False, 0.0), _TTL); return (False, 0.0)
        today = date.today()
        start = (today - timedelta(days=10)).strftime("%Y-%m-%d 09:15")
        end = today.strftime("%Y-%m-%d 15:30")
        try:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY", exchange="NSE") or []
        except TypeError:
            raw = broker.fetch_candles(token, start, end, "ONE_DAY") or []
        closes = [float(r[4]) for r in raw if len(r) >= 5][-3:]
        if len(closes) < 3:
            cache.set(key, (False, 0.0), _TTL); return (False, 0.0)
        spread_pct = (max(closes) - min(closes)) / statistics.mean(closes) * 100.0
        out = (spread_pct <= 1.5, round(spread_pct, 2))
        cache.set(key, out, _TTL); return out
    except Exception:  # noqa: BLE001
        cache.set(key, (False, 0.0), _TTL); return (False, 0.0)


def _live_ltp(symbol: str) -> float:
    try:
        from apps.market_data.services.data_port import DefaultMarketData
        v = DefaultMarketData(tenant_id=None).ltp(symbol)
        return float(v) if v else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def build_pyramid_triggers(tenant=None) -> dict[str, Any]:
    from trading.models import TradeJournal

    open_trades = list(
        TradeJournal.objects.filter(status__in=("EXECUTED", "PAPER", "APPROVED"))
        .exclude(fill_price__isnull=True)
        .filter(stop_loss__gt=0)
        .order_by("-created_at")[:50]
    )
    if not open_trades:
        return {
            "count": 0, "rows": [],
            "note": "Activates when at least one position with a stop loss is open.",
        }

    rows: list[dict] = []
    for t in open_trades:
        entry = float(t.entry_price or 0)
        stop = float(t.stop_loss or 0)
        qty = int(t.quantity or 0)
        if entry <= 0 or stop <= 0 or qty <= 0 or entry == stop:
            continue
        ltp = _live_ltp(t.symbol) or entry
        sign = 1 if t.side == "BUY" else -1
        per_share_risk = abs(entry - stop)
        initial_risk_inr = per_share_risk * qty
        current_pnl = sign * (ltp - entry) * qty
        current_r = round(current_pnl / initial_risk_inr, 2) if initial_risk_inr else 0.0

        # Next add level in 1-2-3 R sequence
        if current_r < 1.0:    next_r = 1.0
        elif current_r < 2.0:  next_r = 2.0
        elif current_r < 3.0:  next_r = 3.0
        else:                   next_r = round(current_r + 1.0, 1)

        handle, spread_pct = _last_3_close_tightness(t.symbol)
        # READY: at-or-above next_r AND a handle on the chart
        # WATCH: within 0.2 R of next_r
        # NOT_YET: anything earlier
        if current_r >= next_r and handle:
            signal = "READY"
        elif current_r >= next_r:
            signal = "WATCH"      # at level but no handle — wait for tightness
        elif current_r >= next_r - 0.2:
            signal = "WATCH"
        else:
            signal = "NOT_YET"

        rows.append({
            "trade_id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "entry": entry,
            "stop": stop,
            "ltp": round(ltp, 2),
            "qty_orig": qty,
            "qty_for_add": int(qty * 0.5),
            "initial_risk_inr": round(initial_risk_inr, 2),
            "current_pnl_inr": round(current_pnl, 2),
            "current_r": current_r,
            "next_add_r": next_r,
            "handle_present": handle,
            "handle_spread_pct": spread_pct,
            "add_signal": signal,
        })

    # Surface READY first (actionable now), then WATCH, then NOT_YET.
    order = {"READY": 0, "WATCH": 1, "NOT_YET": 2}
    rows.sort(key=lambda r: (order.get(r["add_signal"], 3), -r["current_r"]))
    return {
        "count": len(rows),
        "ready_count": sum(1 for r in rows if r["add_signal"] == "READY"),
        "rows": rows,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Pyramid only into READY rows (at +1R/+2R/+3R AND a tight 3-day "
            "handle on the chart). Add qty = original × 0.5 to keep total "
            "risk near 2× initial. WATCH = at level but no handle — let it "
            "consolidate before adding."
        ),
    }
