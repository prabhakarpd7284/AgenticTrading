"""Intraday capital rotation summary.

For a given trading session, bucket every TradeJournal entry into 5-min
slots from 09:15 onward and surface:

  deployed   sum of (qty × entry) for trades that were open in this bucket
  gross      |deployed| — same thing without the sign, for sizing context
  realised_pnl    sum of P&L of trades closed in this bucket
  idle_pct        fraction of capital NOT deployed in this bucket
  sector_breakdown { sector → deployed }
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from typing import Any


_SLOT_MIN = 5
_OPEN = time(9, 15)
_CLOSE = time(15, 30)

# Reuse the sector classifier from correlation_matrix.
from apps.portfolio.services.correlation_matrix import _classify_sector


def _capital() -> float:
    try:
        from trading.models import PortfolioSnapshot
        return float(PortfolioSnapshot.objects.latest().capital)
    except Exception:  # noqa: BLE001
        return 500_000.0


def _slot_for(t: datetime) -> str:
    """Snap to nearest 5-min HH:MM string."""
    minute = (t.minute // _SLOT_MIN) * _SLOT_MIN
    return t.replace(minute=minute, second=0, microsecond=0).strftime("%H:%M")


def _slot_iter(target: date):
    """Yield every HH:MM string from 09:15 to 15:30 in 5-min steps."""
    t = datetime.combine(target, _OPEN)
    end = datetime.combine(target, _CLOSE)
    while t <= end:
        yield t.strftime("%H:%M")
        t += timedelta(minutes=_SLOT_MIN)


def build_intraday_rotation(tenant=None, *, on: str | None = None) -> dict[str, Any]:
    """Aggregate the day's trades into 5-min buckets."""
    from trading.models import TradeJournal

    if on:
        try:
            target = date.fromisoformat(on)
        except ValueError:
            target = date.today()
    else:
        target = date.today()

    trades = list(TradeJournal.objects.filter(trade_date=target))
    capital = _capital()

    buckets: dict[str, dict] = {
        slot: {"slot": slot, "deployed": 0.0, "gross": 0.0,
                "realised_pnl": 0.0, "trades_open": 0, "sector_breakdown": defaultdict(float)}
        for slot in _slot_iter(target)
    }

    for t in trades:
        notional = float(t.entry_price or 0) * int(t.quantity or 0)
        if notional <= 0 or not t.created_at:
            continue
        open_slot = _slot_for(t.created_at.astimezone())
        sector = _classify_sector(t.symbol)

        # An open trade contributes deployed/gross to every bucket from
        # open_slot up to (but not including) the bucket it was filled in.
        # Without explicit exit time we approximate exit = last-snapshot-date.
        fill_slot = _slot_for(t.updated_at.astimezone()) if t.updated_at else "15:30"
        in_run = False
        for slot in buckets:
            if slot == open_slot:
                in_run = True
            if in_run:
                buckets[slot]["deployed"] += notional
                buckets[slot]["gross"] += notional
                buckets[slot]["trades_open"] += 1
                buckets[slot]["sector_breakdown"][sector] += notional
                if slot == fill_slot:
                    if t.fill_price is not None:
                        buckets[slot]["realised_pnl"] += float(t.pnl or 0)
                    in_run = False

    rows = []
    for slot in _slot_iter(target):
        b = buckets[slot]
        idle_pct = max(0.0, 1.0 - (b["deployed"] / capital)) * 100.0 if capital > 0 else 0.0
        rows.append({
            "slot": slot,
            "deployed": round(b["deployed"], 2),
            "gross": round(b["gross"], 2),
            "realised_pnl": round(b["realised_pnl"], 2),
            "trades_open": b["trades_open"],
            "idle_pct": round(idle_pct, 1),
            "sector_breakdown": {k: round(v, 2) for k, v in b["sector_breakdown"].items()},
        })

    peak = max((r["deployed"] for r in rows), default=0.0)
    return {
        "date": target.isoformat(),
        "capital": capital,
        "peak_deployed": round(peak, 2),
        "peak_utilisation_pct": round((peak / capital) * 100, 1) if capital > 0 else 0.0,
        "total_realised_pnl": round(sum(r["realised_pnl"] for r in rows), 2),
        "buckets": rows,
        "note": (
            "5-min buckets from 09:15 to 15:30 IST. Deployed = "
            "capital tied up in open positions in that slot. idle_pct "
            "above 70% suggests under-utilisation; below 20% means leverage "
            "risk during the next adverse move."
        ),
    }
