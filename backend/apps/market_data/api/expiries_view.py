"""Expiries endpoint — enumerates option expiries for an underlying.

GET /api/v1/market-data/expiries/?underlying=NIFTY[&limit=20]

Returns the list straight from the cached Angel scrip master with three
useful pieces of metadata per entry:

  - `expiry`     canonical DDMMMYYYY (e.g. "28MAY2026") — the same key
                  every adapter and the chain endpoint accept.
  - `dte`        days-to-expiry from today (0 on expiry day, negative
                  for past series the master hasn't pruned yet — those
                  are filtered out).
  - `is_weekly`  true when the expiry is on a Tuesday (NIFTY) or Thursday
                  (SENSEX) and not the final week of the month.
  - `is_monthly` true when the expiry is the last weekly of the month
                  (NIFTY post-2025 monthly convention).

The frontend uses this to populate the Options Desk's "Expiry" selector
and label each entry as Weekly / Monthly with DTE inline.
"""
from __future__ import annotations

import logging
from datetime import date, datetime

from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.market_data.services.scrip_master import list_expiries

logger = logging.getLogger(__name__)


def _classify(expiries: list[str]) -> list[dict]:
    """Annotate each expiry with DTE, weekly/monthly flags.

    Monthly = the LAST expiry in a given calendar month (matches both
    NIFTY's last-Tuesday and SENSEX's last-Thursday conventions).
    """
    today = date.today()
    out: list[dict] = []
    # Group by (year, month) so we know which expiry is the "last" in
    # each month — that one gets is_monthly=True.
    by_month: dict[tuple[int, int], list[date]] = {}
    parsed: list[tuple[str, date]] = []
    for e in expiries:
        try:
            d = datetime.strptime(e, "%d%b%Y").date()
        except ValueError:
            continue
        if d < today:
            continue  # filter out stale past expiries
        parsed.append((e, d))
        by_month.setdefault((d.year, d.month), []).append(d)
    last_in_month = {(y, m): max(ds) for (y, m), ds in by_month.items()}
    for raw, d in parsed:
        is_monthly = last_in_month.get((d.year, d.month)) == d
        out.append({
            "expiry": raw,
            "dte": (d - today).days,
            "is_weekly": not is_monthly,
            "is_monthly": is_monthly,
            "weekday": d.strftime("%A"),
        })
    out.sort(key=lambda r: r["dte"])
    return out


class ExpiriesView(APIView):
    """GET /api/v1/market-data/expiries/

    Query params:
      underlying   NIFTY | BANKNIFTY | SENSEX | FINNIFTY  (default NIFTY)
      limit        int — cap on number of entries returned (default 20)
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        underlying = (request.query_params.get("underlying") or "NIFTY").upper()
        try:
            limit = int(request.query_params.get("limit", 20))
        except (TypeError, ValueError):
            limit = 20
        limit = max(1, min(limit, 200))

        try:
            raw = list_expiries(underlying)
        except Exception as e:  # noqa: BLE001
            logger.warning("expiries.scrip_master_failed underlying=%s err=%s",
                            underlying, e)
            raw = []

        entries = _classify(raw)[:limit]
        return Response({
            "underlying": underlying,
            "count": len(entries),
            "expiries": entries,
        })
