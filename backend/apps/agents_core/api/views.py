from functools import lru_cache

from jsonschema import Draft7Validator
from rest_framework import mixins, serializers, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.agents_core.models import AgentRun, AgentStep
from apps.agents_core.registry import strategy_registry
from apps.agents_core.tasks.run import execute_run


@lru_cache(maxsize=64)
def _validator_for(strategy_name: str, strategy_version: str) -> Draft7Validator:
    """Compiled JSONSchema validator per strategy. Compiling on every POST
    burns ~1–5ms; strategies change only at boot (entry-point reload), so
    the (name, version) pair is a stable cache key. Bounded at 64 — well
    above the current 7-strategy ceiling."""
    schema = strategy_registry.get(strategy_name).schema().params or {}
    return Draft7Validator(schema)


class AgentRunSerializer(serializers.ModelSerializer):
    class Meta:
        model = AgentRun
        fields = [
            "id", "strategy_name", "strategy_version", "portfolio",
            "config", "status", "result", "error", "created_at",
            "started_at", "completed_at",
        ]
        read_only_fields = ["id", "status", "result", "error",
                            "created_at", "started_at", "completed_at",
                            "strategy_version"]


# A handful of KPI keys worth surfacing in the run list (across strategies).
_SUMMARY_KEYS = ("realized_pnl_inr", "trades", "won", "total_pnl_inr", "roi_pct", "peak_lots")


class AgentRunListSerializer(serializers.ModelSerializer):
    """Light row for the runs list — NO config/result blobs (those can be huge,
    e.g. a scalp run's full decision log). A small `summary` is derived from the
    result KPIs so the list stays scannable without shipping the whole payload."""
    summary = serializers.SerializerMethodField()

    class Meta:
        model = AgentRun
        fields = ["id", "strategy_name", "strategy_version", "status",
                  "created_at", "started_at", "completed_at", "summary"]

    def get_summary(self, obj):
        result = obj.result if isinstance(obj.result, dict) else {}
        kpis = result.get("kpis")
        if isinstance(kpis, dict):
            out = {k: kpis[k] for k in _SUMMARY_KEYS if k in kpis}
            return out or None
        return None


class AgentRunViewSet(mixins.CreateModelMixin,
                      mixins.ListModelMixin,
                      mixins.RetrieveModelMixin,
                      viewsets.GenericViewSet):
    serializer_class = AgentRunSerializer

    def get_serializer_class(self):
        # List ships the light row; retrieve/create ship the full record.
        return AgentRunListSerializer if self.action == "list" else AgentRunSerializer

    def get_queryset(self):
        qs = AgentRun.objects.filter(tenant=self.request.tenant)
        # Optional filters (cursor pagination already orders by -created_at; the
        # tenant+status / tenant+strategy_name indexes back these filters).
        strategy = self.request.query_params.get("strategy")
        run_status = self.request.query_params.get("status")
        if strategy:
            qs = qs.filter(strategy_name=strategy)
        if run_status:
            qs = qs.filter(status=run_status)
        if self.action == "list":
            # Don't drag config/error over the wire for the list; result is kept
            # only to derive the small `summary` (retrieve loads everything).
            qs = qs.only(
                "id", "tenant_id", "strategy_name", "strategy_version", "status",
                "result", "created_at", "started_at", "completed_at",
            )
        return qs

    def create(self, request, *args, **kwargs):
        ser = self.get_serializer(data=request.data)
        ser.is_valid(raise_exception=True)

        # Resolve the strategy from the plugin registry. Unknown name → 400
        # with the actual list of registered strategies so the operator can
        # see exactly what's available. Catch both KeyError (legacy) and
        # LookupError (current PluginRegistry implementation).
        try:
            strat = strategy_registry.get(ser.validated_data["strategy_name"])
        except (KeyError, LookupError):
            raise serializers.ValidationError({
                "strategy_name": (
                    f"Unknown strategy '{ser.validated_data['strategy_name']}'. "
                    f"Registered: {strategy_registry.names()}"
                ),
            })

        # Validate the config against the strategy's JSONSchema BEFORE
        # enqueueing the Celery task. Without this, a missing required
        # param surfaces as a KeyError deep inside the LangGraph executor —
        # status=failed with a one-word error like ('engine') and the
        # operator has to dig through logs. Validator is cached per
        # (name, version) so this is ~free after the first hit.
        config = ser.validated_data.get("config", {})
        errors = sorted(
            _validator_for(strat.name, strat.version).iter_errors(config),
            key=lambda e: e.path,
        )
        if errors:
            raise serializers.ValidationError({
                "config": [
                    {
                        "path": list(e.absolute_path),
                        "message": e.message,
                    }
                    for e in errors
                ],
            })

        try:
            run = AgentRun.objects.create(
                tenant=request.tenant,
                triggered_by=request.user,
                strategy_name=strat.name,
                strategy_version=strat.version,
                portfolio=ser.validated_data["portfolio"],
                config=config,
            )
        except Exception as exc:  # noqa: BLE001
            # DB error (FK constraint, unique, etc.) — surface as 400 with
            # the exception class + message so the UI's parseDrfError can
            # render it inline next to the form. Without this, the response
            # is Django's HTML 500 page which the React client can't show.
            return Response(
                {"detail": f"Could not create AgentRun: {type(exc).__name__}: {exc}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            execute_run.delay(str(run.id))
        except Exception as exc:  # noqa: BLE001
            # Celery enqueue failed (broker down etc.). The run row exists;
            # mark it so the UI shows the right state and surface the error.
            run.status = AgentRun.Status.FAILED if hasattr(AgentRun, "Status") else "failed"
            run.error = f"enqueue_failed: {type(exc).__name__}: {exc}"
            try:
                run.save(update_fields=["status", "error"])
            except Exception:  # noqa: BLE001
                pass
            return Response(
                self.get_serializer(run).data | {"detail": run.error},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        return Response(self.get_serializer(run).data, status=status.HTTP_202_ACCEPTED)

    @action(detail=True, methods=["get"], url_path="steps")
    def steps(self, request, pk=None):
        """GET /api/v1/agents/runs/{id}/steps/

        Returns the full ordered list of AgentStep rows the run has
        persisted so far. The WebSocket only streams events from the
        moment a client subscribes; the steps endpoint exists so the
        Agent Console can hydrate after a refresh / late navigation
        and show the historical timeline even when the run has finished.

        Response shape mirrors AgentEvent (the wire format used over the
        WS) so the frontend can append-merge both streams keyed by `seq`.
        """
        try:
            run = AgentRun.objects.get(
                tenant=request.tenant, pk=pk,
            )
        except AgentRun.DoesNotExist:
            return Response({"detail": "run not found"}, status=status.HTTP_404_NOT_FOUND)
        rows = (
            AgentStep.objects
            .filter(run=run)
            .order_by("seq")
            .values("seq", "node", "event_type", "payload", "created_at")
        )
        return Response({
            "run_id": str(run.id),
            "count": len(rows),
            "events": [
                {
                    "seq": r["seq"],
                    "node": r["node"],
                    "type": r["event_type"],
                    "payload": r["payload"],
                    "ts": r["created_at"].isoformat(),
                }
                for r in rows
            ],
        })


class StrategyCatalogViewSet(viewsets.ViewSet):
    def list(self, request):
        data = []
        for name, strat in strategy_registry.items():
            s = strat.schema()
            data.append({
                "name": s.name, "version": s.version,
                "asset_class": s.asset_class,
                "params": s.params,
                "required_retrievers": s.required_retrievers,
            })
        return Response(data)
