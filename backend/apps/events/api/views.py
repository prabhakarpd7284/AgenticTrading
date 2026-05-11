from rest_framework import serializers, viewsets
from rest_framework.permissions import IsAuthenticated

from apps.events.models import Event


class EventSerializer(serializers.ModelSerializer):
    class Meta:
        model = Event
        fields = [
            "id", "ts", "type", "severity", "actor_kind", "actor_user",
            "workflow_run", "step_name",
            "trade_id", "order", "signal_id",
            "payload", "text", "request_id",
        ]


class EventViewSet(viewsets.ReadOnlyModelViewSet):
    """Read-only Event log. The unified journal/audit feed.

    Filters via query params:
      ?type=workflow.step.completed
      ?type__startswith=llm.
      ?workflow_run=<uuid>
      ?trade_id=<uuid>
      ?severity=error
      ?actor_kind=workflow
      ?since=2026-05-01           (ts >= since)
      ?until=2026-05-31           (ts <= until)
      ?step_name=plan

    Results paginate via the default DRF pagination.
    """
    serializer_class = EventSerializer
    permission_classes = [IsAuthenticated]

    FILTERABLE = {
        "type", "severity", "actor_kind", "workflow_run", "step_name",
        "trade_id", "signal_id", "request_id",
    }

    def get_queryset(self):
        qs = Event.objects.filter(tenant=self.request.tenant).order_by("-ts")
        params = self.request.query_params

        # Exact-match filters
        for key in self.FILTERABLE:
            v = params.get(key)
            if v is not None and v != "":
                qs = qs.filter(**{key: v})

        # Type prefix (e.g. ?type__startswith=llm.)
        prefix = params.get("type__startswith")
        if prefix:
            qs = qs.filter(type__startswith=prefix)

        # Time window
        since = params.get("since")
        if since:
            qs = qs.filter(ts__gte=since)
        until = params.get("until")
        if until:
            qs = qs.filter(ts__lte=until)

        return qs
