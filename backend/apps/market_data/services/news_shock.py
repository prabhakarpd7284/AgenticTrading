"""News-Shock & Trading-Halt monitor.

When wired to a real NSE corporate-action feed (corpactions.nseindia.com)
this surfaces per-held-symbol shocks (LULD halts, circuit hits, sudden
headlines) and a flatten_recommendation flag.

Until that feed is integrated, this returns an empty list with a clear
note so the React panel renders an actionable "no active shocks" state
instead of an error. The structure is stable so the FE never breaks
when the feed is plugged in.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_news_shock(tenant=None) -> dict[str, Any]:
    """Stub — returns the shape future-us will populate.

    The contract is fixed:
      events: [
        { symbol, severity ∈ {info,warning,critical},
          source, headline, ts, flatten_recommendation: bool }
      ]
    """
    # Held / watchlisted symbols — used to pre-filter once a real feed lands.
    try:
        from trading.models import TradeJournal, WatchlistEntry
        held = set(TradeJournal.objects.filter(
            status__in=("EXECUTED", "PAPER", "APPROVED")
        ).values_list("symbol", flat=True))
        watched = set(WatchlistEntry.objects.values_list("symbol", flat=True))
        coverage = sorted({s for s in (held | watched) if s})[:50]
    except Exception:  # noqa: BLE001
        coverage = []

    return {
        "events": [],
        "coverage_symbols": coverage,
        "as_of": datetime.now(timezone.utc).isoformat(),
        "data_source": "stub",
        "note": (
            "No active shocks. Endpoint contract is stable; populate the "
            "events array when the NSE corporate-action feed is wired into "
            "apps/market_data/services/news_shock.py (poll "
            "https://www.nseindia.com/api/corporates-pit?index=equities&...)."
        ),
    }
