from rest_framework import serializers, viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.strategies.models import Backtest, StrategyInstance
from apps.strategies.services.base_quality import build_base_quality


class StrategyInstanceSerializer(serializers.ModelSerializer):
    class Meta:
        model = StrategyInstance
        fields = "__all__"


class BacktestSerializer(serializers.ModelSerializer):
    class Meta:
        model = Backtest
        fields = "__all__"


class StrategyInstanceViewSet(viewsets.ModelViewSet):
    serializer_class = StrategyInstanceSerializer

    def get_queryset(self):
        return StrategyInstance.objects.filter(tenant=self.request.tenant)

    def perform_create(self, serializer):
        serializer.save(tenant=self.request.tenant)


class BacktestViewSet(viewsets.ModelViewSet):
    serializer_class = BacktestSerializer

    def get_queryset(self):
        return Backtest.objects.filter(tenant=self.request.tenant)

    def perform_create(self, serializer):
        from apps.strategies.tasks.backtest import run_backtest
        bt = serializer.save(tenant=self.request.tenant)
        run_backtest.delay(str(bt.id))


class BaseQualityView(APIView):
    """GET /api/v1/strategies/base-quality/?symbols=A,B,C
    (or no params to score the legacy watchlist).
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        raw = request.query_params.get("symbols") or request.query_params.get("symbol") or ""
        syms = [s.strip().upper() for s in raw.split(",") if s.strip()] if raw else None
        return Response(build_base_quality(symbols=syms))
