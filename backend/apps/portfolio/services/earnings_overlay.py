"""Earnings + Dividend calendar overlay on open positions.

Real-data path: scrape NSE's `/api/event-calendar` for earnings + ex-div
dates per symbol over the next 30 days. NSE rate-limits this hard, so
this stub returns an empty `events` list per position with a clear note,
plus a `coverage_symbols` list so the React panel shows what *would* be
monitored once the feed lands.

Until then, the trader can manually plug earnings dates per symbol via
the planned `WatchlistEntry.earnings_date` field (future work).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_earnings_overlay(tenant=None) -> dict[str, Any]:
    rows: list[dict] = []
    try:
        from trading.models import TradeJournal
        open_trades = TradeJournal.objects.filter(
            status__in=("EXECUTED", "PAPER", "APPROVED")
        ).exclude(fill_price__isnull=True)
        for t in open_trades[:50]:
            rows.append({
                "trade_id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "qty": int(t.quantity or 0),
                "earnings_date": None,
                "ex_div_date": None,
                "consensus_eps": None,
                "avg_post_earn_gap_pct": None,
                "days_to_event": None,
            })
    except Exception:  # noqa: BLE001
        pass

    return {
        "count": len(rows),
        "rows": rows,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "data_source": "stub",
        "note": (
            "Earnings + dividend fields are null until the NSE event-calendar "
            "feed is wired into apps/portfolio/services/earnings_overlay.py. "
            "Contract is stable — the React panel renders a row per open "
            "position so the wiring becomes a pure data swap."
        ),
    }
