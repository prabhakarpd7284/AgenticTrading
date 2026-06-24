from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import (
    OpenApiParameter, extend_schema, extend_schema_view,
)
from rest_framework import serializers, viewsets

from apps.strategies.models import Backtest, StrategyInstance

# Watchlist/StrategyInstance/Backtest all carry UUID primary keys. Pin the
# detail-route {id} param to UUID so the generated schema doesn't default it
# to "string" (the cause of the "could not derive type of path parameter"
# warning). Annotation only — routing + lookup_field are untouched.
_UUID_PK = [OpenApiParameter("id", OpenApiTypes.UUID, OpenApiParameter.PATH)]
_uuid_detail_schema = extend_schema_view(
    retrieve=extend_schema(parameters=_UUID_PK),
    update=extend_schema(parameters=_UUID_PK),
    partial_update=extend_schema(parameters=_UUID_PK),
    destroy=extend_schema(parameters=_UUID_PK),
)


class StrategyInstanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = StrategyInstance
        fields = "__all__"


class BacktestSerializer(serializers.ModelSerializer):
    class Meta:
        model = Backtest
        fields = "__all__"


@_uuid_detail_schema
class StrategyInstanceViewSet(viewsets.ModelViewSet):
    serializer_class = StrategyInstanceSerializer

    def get_queryset(self):
        return StrategyInstance.objects.filter(tenant=self.request.tenant)

    def perform_create(self, serializer):
        serializer.save(tenant=self.request.tenant)


@_uuid_detail_schema
class BacktestViewSet(viewsets.ModelViewSet):
    serializer_class = BacktestSerializer

    def get_queryset(self):
        return Backtest.objects.filter(tenant=self.request.tenant)

    def perform_create(self, serializer):
        from apps.strategies.tasks.backtest import run_backtest
        bt = serializer.save(tenant=self.request.tenant)
        run_backtest.delay(str(bt.id))
