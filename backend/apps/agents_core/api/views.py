from functools import lru_cache

from jsonschema import Draft7Validator
from rest_framework import mixins, serializers, status, viewsets
from rest_framework.response import Response

from apps.agents_core.models import AgentRun
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


class AgentRunViewSet(mixins.CreateModelMixin,
                      mixins.ListModelMixin,
                      mixins.RetrieveModelMixin,
                      viewsets.GenericViewSet):
    serializer_class = AgentRunSerializer

    def get_queryset(self):
        return AgentRun.objects.filter(tenant=self.request.tenant)

    def create(self, request, *args, **kwargs):
        ser = self.get_serializer(data=request.data)
        ser.is_valid(raise_exception=True)

        # Resolve the strategy from the plugin registry. Unknown name → 400
        # with a useful "available strategies" hint instead of a 500.
        try:
            strat = strategy_registry.get(ser.validated_data["strategy_name"])
        except KeyError:
            raise serializers.ValidationError({
                "strategy_name": (
                    f"Unknown strategy. Available: "
                    f"{sorted(strategy_registry.keys())}"
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

        run = AgentRun.objects.create(
            tenant=request.tenant,
            triggered_by=request.user,
            strategy_name=strat.name,
            strategy_version=strat.version,
            portfolio=ser.validated_data["portfolio"],
            config=config,
        )
        execute_run.delay(str(run.id))
        return Response(self.get_serializer(run).data, status=status.HTTP_202_ACCEPTED)


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
