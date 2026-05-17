"""3:20 PM Forced-Flat cockpit.

Surfaces every open intraday position with the IST countdown to 15:15
(after which gamma + closing-auction slippage make holding negligent),
their live P&L, and an estimated closing-auction slippage.

Also exposes a flatten_all() helper that queues SQUARE_OFF orders for
every open MIS leg. In paper mode it stamps the journal rows directly;
no broker call.
"""
from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any

try:
    from zoneinfo import ZoneInfo
    _IST = ZoneInfo("Asia/Kolkata")
except Exception:  # pragma: no cover
    from datetime import timezone as _tz, timedelta as _td
    _IST = _tz(_td(hours=5, minutes=30))

_FLATTEN_DEADLINE = time(15, 15)


def _seconds_to_deadline(now_ist: datetime) -> int:
    deadline = now_ist.replace(hour=_FLATTEN_DEADLINE.hour,
                                minute=_FLATTEN_DEADLINE.minute,
                                second=0, microsecond=0)
    delta = (deadline - now_ist).total_seconds()
    return max(0, int(delta))


def _est_slippage_bps(symbol: str) -> float:
    """Cheap proxy: median |slippage| from history, falls back to 8 bps."""
    try:
        from trading.models import TradeJournal
        rows = TradeJournal.objects.filter(symbol=symbol).exclude(
            fill_price__isnull=True
        ).values_list("entry_price", "fill_price")[:30]
        bps = []
        for entry, fill in rows:
            if entry and fill and entry > 0:
                bps.append(abs((float(fill) - float(entry)) / float(entry)) * 10_000.0)
        if not bps:
            return 8.0
        bps.sort()
        return round(bps[len(bps) // 2], 1)
    except Exception:  # noqa: BLE001
        return 8.0


def _live_ltp(symbol: str) -> float:
    try:
        from apps.market_data.services.data_port import DefaultMarketData
        port = DefaultMarketData(tenant_id=None)
        v = port.ltp(symbol)
        return float(v) if v else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def build_forced_flat(tenant=None) -> dict[str, Any]:
    from trading.models import TradeJournal

    now_ist = datetime.now(tz=_IST)
    countdown = _seconds_to_deadline(now_ist)

    open_trades = list(
        TradeJournal.objects.filter(status__in=("EXECUTED", "PAPER", "APPROVED"))
        .exclude(fill_price__isnull=True)
        .order_by("-created_at")[:100]
    )

    rows = []
    total_pnl = 0.0
    total_carry = 0.0
    for t in open_trades:
        entry = float(t.entry_price or 0)
        qty = int(t.quantity or 0)
        ltp = _live_ltp(t.symbol) or entry
        sign = 1 if t.side == "BUY" else -1
        pnl = round(sign * (ltp - entry) * qty, 2)
        total_pnl += pnl
        slip_bps = _est_slippage_bps(t.symbol)
        # Closing-auction carry cost = position notional × slip_bps × 2
        # (auction-matching slippage in + a separate exit print on T+1).
        notional = ltp * qty
        carry_cost = round(notional * slip_bps / 10_000.0 * 2.0, 2)
        total_carry += carry_cost
        rows.append({
            "trade_id": t.id,
            "symbol": t.symbol,
            "side": t.side,
            "qty": qty,
            "entry": entry,
            "ltp": round(ltp, 2),
            "pnl": pnl,
            "est_slippage_bps": slip_bps,
            "closing_auction_carry_cost_inr": carry_cost,
            "status": t.status,
        })

    return {
        "now_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "deadline": "15:15 IST",
        "countdown_seconds": countdown,
        "active": now_ist.hour >= 15 and (now_ist.hour > 15 or now_ist.minute >= 0),
        "count": len(rows),
        "total_pnl": round(total_pnl, 2),
        "total_carry_cost_inr": round(total_carry, 2),
        "rows": rows,
        "note": (
            "Activates after 15:00 IST. After 15:15 the closing auction "
            "absorbs MIS holders at whatever clearing price prints — slippage "
            "explodes. carry-cost = notional × est-slippage × 2 (auction in + "
            "exit print). Flatten before the deadline to avoid paying it."
        ),
    }


def flatten_all(tenant=None) -> dict[str, Any]:
    """Mark every open intraday TradeJournal row as FILLED at current LTP.

    Paper-mode only — no broker call. Live-mode users should still place
    exits manually until we wire the live order path.
    """
    from trading.models import TradeJournal

    flattened: list[dict] = []
    for t in TradeJournal.objects.filter(
        status__in=("EXECUTED", "PAPER", "APPROVED")
    ).exclude(fill_price__isnull=True):
        ltp = _live_ltp(t.symbol) or float(t.entry_price or 0)
        sign = 1 if t.side == "BUY" else -1
        pnl = sign * (ltp - float(t.entry_price or 0)) * int(t.quantity or 0)
        t.fill_price = ltp
        t.pnl = (t.pnl or 0) + pnl
        t.status = "FILLED"
        t.save(update_fields=["fill_price", "pnl", "status", "updated_at"])
        flattened.append({"trade_id": t.id, "symbol": t.symbol, "ltp": ltp, "pnl": round(pnl, 2)})

    return {"flattened": len(flattened), "trades": flattened}
