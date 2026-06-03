"""Combined positions endpoint — aggregates across every active BrokerLink for the tenant.

Reads BrokerPositionSnapshot rows (written by the Celery refresh task or
on-demand by /brokers/{id}/refresh/). Optionally triggers a synchronous
refresh first via ``?refresh=1``.

Response shape (UI contract):
  {
    "fetched_at": "2026-05-18T12:34:56+05:30",   # max snapshot ts across links
    "stalest_age_seconds": 12.4,
    "brokers": [
      {
        "link_id": "...",
        "broker_name": "angel_one",
        "display_name": "Personal",
        "status": "active",
        "is_default": true,
        "fetched_at": "...",
        "age_seconds": 12.4,
        "ok": true,
        "error": "",
        "positions": [...],
        "holdings": [...],
        "margin": {...},
      },
      ...
    ],
    "totals": {
      "positions_count": 7,
      "holdings_count": 12,
      "open_pnl": 1234.50,
      "available_cash": 250000.0,
      "used_margin": 50000.0,
    },
  }
"""
from __future__ import annotations

from django.utils import timezone
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.market_data.api.broker_views import (
    DAILY_TOKEN_BROKERS, _take_snapshot, _token_valid_today,
)
from apps.market_data.models import BrokerLink


class CombinedPositionsView(APIView):
    """GET /api/v1/positions/combined/ — multi-broker aggregated view."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        force_refresh = request.query_params.get("refresh") in ("1", "true")
        only_status = request.query_params.get("status")  # 'active' to skip errored

        qs = BrokerLink.objects.filter(tenant=request.tenant)
        if only_status:
            qs = qs.filter(status=only_status)
        links = list(qs.select_related("tenant"))

        if force_refresh:
            for link in links:
                try:
                    _take_snapshot(link)
                except Exception:
                    pass

        brokers = []
        max_age_seconds = 0.0
        latest_fetched = None
        total_open_pnl = 0.0
        total_cash = 0.0
        total_used = 0.0
        total_positions = 0
        total_holdings = 0
        now = timezone.now()

        for link in links:
            snap = link.snapshots.first()
            if snap is None:
                brokers.append({
                    "link_id": str(link.id),
                    "broker_name": link.broker_name,
                    "display_name": link.display_name,
                    "status": link.status,
                    "is_default": link.is_default,
                    "fetched_at": None,
                    "age_seconds": None,
                    "ok": None,
                    "error": "no snapshot yet",
                    "positions": [], "holdings": [], "margin": {},
                    "requires_daily_login": link.broker_name in DAILY_TOKEN_BROKERS,
                    "token_valid_today": _token_valid_today(link),
                })
                continue

            age = (now - snap.fetched_at).total_seconds()
            if snap.ok:
                max_age_seconds = max(max_age_seconds, age)
                if latest_fetched is None or snap.fetched_at > latest_fetched:
                    latest_fetched = snap.fetched_at

                total_open_pnl += sum(_f(p.get("pnl")) + _f(p.get("mtm")) for p in snap.positions)
                total_cash += _f((snap.margin or {}).get("available_cash"))
                total_used += _f((snap.margin or {}).get("used"))
                total_positions += len(snap.positions or [])
                total_holdings += len(snap.holdings or [])

            brokers.append({
                "link_id": str(link.id),
                "broker_name": link.broker_name,
                "display_name": link.display_name,
                "status": link.status,
                "is_default": link.is_default,
                "fetched_at": snap.fetched_at.isoformat(),
                "age_seconds": round(age, 1),
                "ok": snap.ok,
                "error": snap.error,
                "positions": snap.positions,
                "holdings": snap.holdings,
                "margin": snap.margin,
                "requires_daily_login": link.broker_name in DAILY_TOKEN_BROKERS,
                "token_valid_today": _token_valid_today(link),
            })

        return Response({
            "fetched_at": latest_fetched.isoformat() if latest_fetched else None,
            "stalest_age_seconds": round(max_age_seconds, 1) if max_age_seconds else None,
            "brokers": brokers,
            "totals": {
                "positions_count": total_positions,
                "holdings_count": total_holdings,
                "open_pnl": round(total_open_pnl, 2),
                "available_cash": round(total_cash, 2),
                "used_margin": round(total_used, 2),
            },
        })


def _f(v) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0
