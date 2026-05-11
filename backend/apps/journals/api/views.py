"""Backward-compat views: /api/v1/journals/ now reads from apps.events.Event.

The legacy `JournalEntry` model is kept (empty) only for migration
history. New writes flow into Event via JournalAdapter. Frontend will
migrate to /api/v1/events/ in the next IA pass.
"""
from rest_framework import serializers, viewsets
from rest_framework.permissions import IsAuthenticated

from apps.events.models import Event


# Event types that correspond to "journal entries" in the legacy sense.
JOURNAL_EVENT_TYPES = [
    Event.Type.TRADE_PLANNED,
    Event.Type.TRADE_OPENED,
    Event.Type.TRADE_CLOSED,
    Event.Type.TRADE_SL_HIT,
    Event.Type.TRADE_TGT_HIT,
    Event.Type.TRADE_TRAILED,
    Event.Type.RISK_APPROVED,
    Event.Type.RISK_REJECTED,
    Event.Type.STRADDLE_LEG_ROLLED,
    Event.Type.STRADDLE_CLOSED,
    Event.Type.PYRAMID_ADD,
    Event.Type.PYRAMID_EXIT,
]


class JournalEntrySerializer(serializers.ModelSerializer):
    # Translate Event field names to the legacy JournalEntry shape so the
    # frontend doesn't have to change yet.
    kind = serializers.CharField(source="type")
    title = serializers.CharField(source="text")
    body = serializers.SerializerMethodField()
    meta = serializers.JSONField(source="payload")

    class Meta:
        model = Event
        fields = ["id", "kind", "title", "body", "meta", "created_at"]

    def get_body(self, obj):
        return (obj.payload or {}).get("body", "")


class JournalEntryViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = JournalEntrySerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return (
            Event.objects
            .filter(tenant=self.request.tenant, type__in=JOURNAL_EVENT_TYPES)
            .order_by("-ts")
        )
