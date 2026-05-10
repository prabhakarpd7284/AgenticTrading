from rest_framework import mixins, serializers, status, viewsets
from rest_framework.response import Response

from apps.agents_core.models import AgentRun
from apps.agents_core.registry import strategy_registry
from apps.agents_core.tasks.run import execute_run


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
        strat = strategy_registry.get(ser.validated_data["strategy_name"])
        run = AgentRun.objects.create(
            tenant=request.tenant,
            triggered_by=request.user,
            strategy_name=strat.name,
            strategy_version=strat.version,
            portfolio=ser.validated_data["portfolio"],
            config=ser.validated_data.get("config", {}),
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
