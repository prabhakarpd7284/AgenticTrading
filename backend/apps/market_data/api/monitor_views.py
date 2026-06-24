"""Broker-call / rate-limit monitoring endpoint for the ops dashboard.

Surfaces the telemetry behind the 2026-06-24 rate-limit incident so an
operator can see, live: the Angel SmartAPI rate-limit circuit breaker state,
per-minute SmartAPI call volume, Celery queue depths (early warning for a
backlog pile-up), and per-broker-link health.
"""
from __future__ import annotations

from django.utils import timezone
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.market_data.models import BrokerLink, BrokerPositionSnapshot

# Celery queues the dev worker consumes (-Q celery,agents,orders,backtests).
_BROKER_QUEUES = ("celery", "orders", "agents", "backtests")


class BrokerMonitorView(APIView):
    """Live broker telemetry: rate-limit breaker, SmartAPI call volume,
    Celery queue depths, and per-link health. Tenant-scoped; poll it."""

    permission_classes = [IsAuthenticated]

    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request):
        from trading.services.data_service import (
            BrokerClient, _get_redis_for_throttle,
        )

        bc = BrokerClient.get_instance()

        # Celery queue depths from the shared :6380 broker (LLEN, best effort).
        queues: dict = {}
        try:
            r = _get_redis_for_throttle()
        except Exception:
            r = None
        if r is not None:
            for q in _BROKER_QUEUES:
                try:
                    queues[q] = int(r.llen(q))
                except Exception:
                    queues[q] = None

        # Per-link health (tenant-scoped) + latest snapshot status.
        links: list = []
        tenant = getattr(request, "tenant", None)
        qs = BrokerLink.objects.all()
        if tenant is not None:
            qs = qs.filter(tenant=tenant)
        for link in qs.exclude(status=BrokerLink.Status.DISABLED):
            snap = (
                BrokerPositionSnapshot.objects.filter(link=link)
                .order_by("-fetched_at")
                .values("ok", "error", "fetched_at")
                .first()
            )
            links.append({
                "id": str(link.id),
                "broker": link.broker_name,
                "display_name": link.display_name or link.broker_name,
                "status": link.status,
                "last_refreshed_at": (
                    link.last_refreshed_at.isoformat() if link.last_refreshed_at else None
                ),
                "last_error": link.last_error or (snap["error"] if snap else ""),
                "last_snapshot_ok": snap["ok"] if snap else None,
            })

        return Response({
            "breaker": bc.breaker_status(),
            "call_rate": bc.recent_call_rate(30),
            "queues": queues,
            "links": links,
            "ts": timezone.now().isoformat(),
        })
