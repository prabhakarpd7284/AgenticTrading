"""TradingView integration endpoints.

Three distinct surfaces:

  TradingViewLinkViewSet — auth'd CRUD for the operator's webhook
  configurations. Mounted under /api/v1/notifications/tradingview/.

  TradingViewWatchlistViewSet — auth'd CRUD for named symbol lists.
  Mounted under /api/v1/notifications/tradingview/watchlists/.

  TradingViewWebhookView — the public endpoint TradingView POSTs to.
  Mounted at /api/v1/webhooks/tradingview/<secret>/ at the URL-conf top
  level. No JWT; the secret in the URL path IS the auth.

  GroupedSignalsView — auth'd read-only aggregator of incoming
  TradingView signals, faceted by symbol / strategy / source / day for
  the TradingView Manager page.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import timedelta
from typing import Literal

import structlog
from django.db.models import Count, Max, Q
from django.utils import timezone
from rest_framework import mixins, permissions, serializers, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.notifications.models import (
    TradingViewLink, TradingViewSignal, TradingViewWatchlist,
)
from apps.notifications.services.tradingview import (
    fire_workflow, parse_payload, record_alert,
)
from apps.strategies.models import Signal

log = structlog.get_logger()


# ── Serializers ──────────────────────────────────────────────────────────

class TradingViewLinkSerializer(serializers.ModelSerializer):
    webhook_url = serializers.SerializerMethodField()

    class Meta:
        model = TradingViewLink
        fields = [
            "id", "display_name", "is_active",
            "autofire_enabled", "default_strategy_name",
            "portfolio", "allowed_actions",
            "webhook_secret", "webhook_url",
            "last_received_at", "receive_count", "last_error",
            "created_at", "updated_at",
        ]
        read_only_fields = [
            "id", "webhook_secret", "webhook_url",
            "last_received_at", "receive_count", "last_error",
            "created_at", "updated_at",
        ]

    def get_webhook_url(self, link: TradingViewLink) -> str:
        request = self.context.get("request")
        path = f"/api/v1/webhooks/tradingview/{link.webhook_secret}/"
        return request.build_absolute_uri(path) if request else path


class TradingViewSignalSerializer(serializers.ModelSerializer):
    class Meta:
        model = TradingViewSignal
        fields = [
            "id", "received_at", "parsed", "parse_error",
            "raw_payload", "signal", "workflow_run",
        ]


# ── CRUD viewset ─────────────────────────────────────────────────────────

class TradingViewLinkViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    serializer_class = TradingViewLinkSerializer
    # Restrict pk regex to UUIDs so the URL conf can host sibling
    # routers like `tradingview/watchlists/` without the detail pattern
    # `tradingview/<pk>/` accidentally swallowing "watchlists" as a pk.
    lookup_value_regex = r"[0-9a-fA-F-]{32,36}"

    def get_queryset(self):
        # Scoped to the requesting user — different traders in the same
        # tenant don't see each other's webhook URLs.
        return TradingViewLink.objects.filter(
            tenant=self.request.tenant,
            owner=self.request.user,
        ).order_by("-created_at")

    def perform_create(self, serializer):
        serializer.save(tenant=self.request.tenant, owner=self.request.user)

    @action(detail=True, methods=["post"], url_path="rotate-secret")
    def rotate_secret(self, request, pk=None):
        link = self.get_object()
        link.rotate_secret()
        return Response(self.get_serializer(link).data)

    @action(detail=True, methods=["get"], url_path="recent")
    def recent(self, request, pk=None):
        link = self.get_object()
        rows = TradingViewSignal.objects.filter(link=link).order_by("-received_at")[:20]
        return Response(TradingViewSignalSerializer(rows, many=True).data)


# ── Public webhook receiver ──────────────────────────────────────────────

class TradingViewWebhookView(APIView):
    """Receives TradingView alerts. Auth is the URL-path secret."""

    permission_classes = [permissions.AllowAny]
    # TradingView servers will retry on 5xx — never 5xx for parse problems;
    # 200 with parse_error in the body is the right answer (the audit row is
    # safely persisted). Reserve 4xx for real auth/lookup failures.

    def post(self, request, secret: str):
        try:
            link = TradingViewLink.objects.get(webhook_secret=secret, is_active=True)
        except TradingViewLink.DoesNotExist:
            return Response(
                {"detail": "Unknown or inactive webhook."},
                status=status.HTTP_404_NOT_FOUND,
            )

        raw = _decode_body(request)
        parsed = None
        parse_error = ""
        try:
            parsed = parse_payload(raw, request.content_type or "")
        except ValueError as exc:
            parse_error = str(exc)
            log.info("tradingview.parse_failed",
                     link_id=str(link.id), error=parse_error)

        tv_signal = record_alert(link, raw, parsed, parse_error)

        run_id = None
        if parsed and not parse_error:
            run_id = fire_workflow(link, parsed)
            if run_id:
                tv_signal.workflow_run_id = run_id
                tv_signal.save(update_fields=["workflow_run"])

        return Response({
            "received": True,
            "id": tv_signal.id,
            "parsed": tv_signal.parsed,
            "parse_error": parse_error or None,
            "run_id": run_id,
        }, status=status.HTTP_200_OK)


def _decode_body(request) -> str:
    """Return the request body as text. TradingView posts UTF-8 either
    application/json or text/plain; in dev a curl test might post bytes."""
    body = request.body
    if isinstance(body, bytes):
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError:
            return body.decode("utf-8", errors="replace")
    return str(body or "")


# ── Watchlists ───────────────────────────────────────────────────────────

class TradingViewWatchlistSerializer(serializers.ModelSerializer):
    symbol_count = serializers.SerializerMethodField()

    class Meta:
        model = TradingViewWatchlist
        fields = [
            "id", "name", "description", "symbols", "symbol_count",
            "created_at", "updated_at",
        ]
        read_only_fields = ["id", "symbol_count", "created_at", "updated_at"]

    def get_symbol_count(self, w: TradingViewWatchlist) -> int:
        return len(w.symbols or [])


class TradingViewWatchlistViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.UpdateModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    serializer_class = TradingViewWatchlistSerializer

    def get_queryset(self):
        return TradingViewWatchlist.objects.filter(
            tenant=self.request.tenant,
            owner=self.request.user,
        )

    def perform_create(self, serializer):
        # Pre-flight uniqueness check — without this, the unique_together DB
        # constraint fires as IntegrityError → 500 instead of a friendly 400.
        name = serializer.validated_data.get("name", "").strip()
        exists = TradingViewWatchlist.objects.filter(
            tenant=self.request.tenant, owner=self.request.user, name=name,
        ).exists()
        if exists:
            raise serializers.ValidationError({
                "name": f"A watchlist named {name!r} already exists.",
            })
        serializer.save(tenant=self.request.tenant, owner=self.request.user)

    @action(detail=True, methods=["post"], url_path="add-symbols")
    def add_symbols(self, request, pk=None):
        """Bulk-add symbols to an existing watchlist. Body: {"symbols": [...]}.
        Normalisation + dedup happens in model.save()."""
        watchlist = self.get_object()
        incoming = request.data.get("symbols") or []
        if not isinstance(incoming, list):
            raise serializers.ValidationError({"symbols": "Must be a list."})
        watchlist.symbols = list(watchlist.symbols or []) + [str(s) for s in incoming]
        watchlist.save()
        return Response(self.get_serializer(watchlist).data)

    @action(detail=True, methods=["post"], url_path="remove-symbols")
    def remove_symbols(self, request, pk=None):
        """Bulk-remove symbols from a watchlist. Body: {"symbols": [...]}."""
        watchlist = self.get_object()
        incoming = request.data.get("symbols") or []
        if not isinstance(incoming, list):
            raise serializers.ValidationError({"symbols": "Must be a list."})
        targets = {str(s).upper().strip() for s in incoming}
        watchlist.symbols = [s for s in (watchlist.symbols or []) if s not in targets]
        watchlist.save()
        return Response(self.get_serializer(watchlist).data)


# ── Grouped signals (TradingView Manager view) ───────────────────────────

GroupBy = Literal["symbol", "strategy", "source", "day"]
_VALID_GROUP_BY: set[str] = {"symbol", "strategy", "source", "day"}


class GroupedSignalsView(APIView):
    """Faceted aggregation of recent Signal rows.

    Query params:
      by=symbol|strategy|source|day   (default: symbol)
      days=N                          (default: 7, max: 90)
      source=TRADINGVIEW|...          (optional — filter to one source)
      symbol=STR                      (optional — filter to one symbol)
      watchlist=<uuid>                (optional — filter to symbols in this watchlist)

    Returns:
      {
        by: "symbol",
        rows: [
          {key: "RELIANCE", count: 12, buys: 8, sells: 4,
           latest_at: "2026-05-19T19:34:02Z", latest_action: "BUY"},
          ...
        ]
      }
    """

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        tenant = request.tenant
        by = (request.query_params.get("by") or "symbol").lower()
        if by not in _VALID_GROUP_BY:
            return Response(
                {"detail": f"by must be one of {sorted(_VALID_GROUP_BY)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            days = min(max(int(request.query_params.get("days") or 7), 1), 90)
        except ValueError:
            days = 7
        since = timezone.now() - timedelta(days=days)

        qs = Signal.objects.filter(tenant=tenant, signal_time__gte=since)

        source = request.query_params.get("source")
        if source:
            qs = qs.filter(source=source.upper())

        symbol = request.query_params.get("symbol")
        if symbol:
            qs = qs.filter(symbol=symbol.upper())

        watchlist_id = request.query_params.get("watchlist")
        if watchlist_id:
            wl = (TradingViewWatchlist.objects
                  .filter(tenant=tenant, owner=request.user, pk=watchlist_id)
                  .first())
            if wl and wl.symbols:
                qs = qs.filter(symbol__in=wl.symbols)
            else:
                qs = qs.none()

        rows = _group_signals(qs, by)
        return Response({"by": by, "rows": rows, "window_days": days})


def _group_signals(qs, by: str) -> list[dict]:
    """Bucket the queryset by the chosen dimension. Returns rows sorted by
    `count desc` (most active first). `day` returns ISO date strings; the
    UI is responsible for any further locale formatting."""

    if by == "day":
        # Postgres has TruncDate; sqlite test path also supports it via Django.
        from django.db.models.functions import TruncDate
        qs = qs.annotate(_day=TruncDate("signal_time"))
        group_field = "_day"
    else:
        group_field = {"symbol": "symbol", "strategy": "strategy", "source": "source"}[by]

    aggregates = {
        "count": Count("id"),
        "latest_at": Max("signal_time"),
        "buys": Count("id", filter=Q(side="BUY")),
        "sells": Count("id", filter=Q(side="SELL")),
    }
    grouped = (
        qs.values(group_field)
          .annotate(**aggregates)
          .order_by("-count", "-latest_at")
    )

    # The latest_action for each bucket needs a second pass — Django can't
    # aggregate "value of column X for row with max date Y" in one query.
    # For small N (typical of a 7-day window) the follow-up is cheap.
    rows: list[dict] = []
    for g in grouped:
        key = g[group_field]
        key_str = key.isoformat() if hasattr(key, "isoformat") else str(key)
        latest = (
            qs.filter(**{group_field: key})
              .order_by("-signal_time")
              .values_list("side", flat=True)
              .first()
        )
        rows.append(OrderedDict([
            ("key", key_str),
            ("count", g["count"]),
            ("buys", g["buys"]),
            ("sells", g["sells"]),
            ("latest_at", g["latest_at"].isoformat() if g["latest_at"] else None),
            ("latest_action", latest or ""),
        ]))
    return rows
