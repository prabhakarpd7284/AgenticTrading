"""Nightly scan REST surface — mounted at /api/v1/nightly-scan/ (read-only).

  GET  ''                     recent scans (date, status, counts)
  GET  latest/                newest finished scan + per-stock rows
  GET  <YYYY-MM-DD>/          scan governing that trading day + rows
  GET  <YYYY-MM-DD>/<SYMBOL>/ one row + every headline for the stock in the window

Scans/headlines are shared reference data (not tenant-scoped); the views
still require an authenticated user with an active tenant like every other
/api/v1 route. Per-portfolio cooldown lives at /api/v1/trading-rules/.
"""
from __future__ import annotations

from datetime import date, timedelta

from django.shortcuts import get_object_or_404
from django.utils import timezone
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema
from rest_framework.exceptions import NotFound, ValidationError
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.common.permissions import TenantScoped
from apps.news.models import NewsItem, NightlyScan, NightlyScanRow
from apps.news.services.sentiment import cfg as sentiment_cfg


def _f(x):
    return None if x is None else float(x)


def row_payload(r: NightlyScanRow) -> dict:
    return {
        "symbol": r.symbol, "data_date": r.data_date.isoformat() if r.data_date else None,
        "levels": r.levels, "cpr_width_pct": _f(r.cpr_width_pct), "cpr_class": r.cpr_class,
        "two_day_relationship": r.two_day_relationship,
        "camarilla_relationship": r.camarilla_relationship, "bias": r.bias,
        "sentiment_score": _f(r.sentiment_score), "sentiment_label": r.sentiment_label,
        "sentiment_weight": _f(r.sentiment_weight),
        "headline_count": r.headline_count, "scored_count": r.scored_count,
        "event_tomorrow": r.event_tomorrow, "event_purpose": r.event_purpose,
        "allow_long": r.allow_long, "allow_short": r.allow_short, "allowed": r.allowed,
        "reasons": r.reasons, "headlines": r.headlines,
    }


def scan_payload(scan: NightlyScan, *, rows: bool = True) -> dict:
    out = {
        "id": scan.id, "for_date": scan.for_date.isoformat(),
        "session_date": scan.session_date.isoformat(), "status": scan.status,
        "started_at": scan.started_at.isoformat() if scan.started_at else None,
        "finished_at": scan.finished_at.isoformat() if scan.finished_at else None,
        "timings": scan.timings, "sources": scan.sources, "summary": scan.summary,
        "telegram_sent": scan.telegram_sent, "error": scan.error,
    }
    if rows:
        out["rows"] = [row_payload(r) for r in scan.rows.all()]
    return out


def _parse_date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError as e:
        raise ValidationError({"date": "expected YYYY-MM-DD"}) from e


class ScanListView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]

    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request):
        scans = NightlyScan.objects.order_by("-for_date")[:30]
        return Response({"results": [
            {"id": s.id, "for_date": s.for_date.isoformat(), "session_date": s.session_date.isoformat(),
             "status": s.status, "restricted": len((s.summary or {}).get("skip", [])),
             "market": (s.summary or {}).get("market", {}).get("label"),
             "telegram_sent": s.telegram_sent, "total_secs": (s.timings or {}).get("total")}
            for s in scans
        ]})


class LatestScanView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]

    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request):
        scan = (NightlyScan.objects.exclude(status=NightlyScan.Status.RUNNING)
                .order_by("-for_date").first())
        running = NightlyScan.objects.filter(status=NightlyScan.Status.RUNNING).exists()
        if scan is None:
            return Response({"scan": None, "running": running,
                             "thresholds": _thresholds()})
        return Response({"scan": scan_payload(scan), "running": running,
                         "thresholds": _thresholds()})


class ScanByDateView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]

    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request, for_date: str):
        scan = get_object_or_404(NightlyScan, for_date=_parse_date(for_date))
        return Response({"scan": scan_payload(scan), "running": scan.status == "running",
                         "thresholds": _thresholds()})


class ScanRowView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]

    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request, for_date: str, symbol: str):
        scan = get_object_or_404(NightlyScan, for_date=_parse_date(for_date))
        row = scan.rows.filter(symbol=symbol.upper()).first()
        if row is None:
            raise NotFound(f"{symbol} not in the scan for {for_date}")
        until = scan.finished_at or timezone.now()
        since = until - timedelta(days=int(sentiment_cfg("LOOKBACK_DAYS")))
        items = (NewsItem.objects.filter(mentions__symbol=row.symbol, published_at__gte=since,
                                         published_at__lte=until + timedelta(hours=1))
                 .distinct().order_by("-published_at")[:100])
        return Response({
            "for_date": scan.for_date.isoformat(), "row": row_payload(row),
            "headlines": [{
                "id": it.id, "title": it.title, "url": it.url, "source": it.source,
                "publisher": it.publisher, "kind": it.kind,
                "published_at": it.published_at.isoformat(), "score": _f(it.sentiment_score),
                "materiality": it.materiality or None, "confidence": _f(it.confidence),
            } for it in items],
        })


def _thresholds() -> dict:
    return {k: sentiment_cfg(k) for k in ("STRONG_THRESHOLD", "MILD_THRESHOLD",
                                          "MIN_STRONG_WEIGHT", "HALF_LIFE_HOURS")}
