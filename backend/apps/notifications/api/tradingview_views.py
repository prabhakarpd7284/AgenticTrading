"""TradingView integration endpoints.

Two distinct surfaces:

  TradingViewLinkViewSet — auth'd CRUD for the operator's webhook
  configurations. Mounted under /api/v1/notifications/tradingview/.

  webhook_receive — the public endpoint TradingView POSTs to. Mounted at
  /api/v1/webhooks/tradingview/<secret>/ at the URL-conf top level. No JWT;
  the secret in the URL path IS the auth.
"""
from __future__ import annotations

import structlog
from rest_framework import mixins, permissions, serializers, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.notifications.models import TradingViewLink, TradingViewSignal
from apps.notifications.services.tradingview import (
    fire_workflow, parse_payload, record_alert,
)

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
