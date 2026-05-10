from rest_framework import serializers, viewsets

from apps.strategies.models import Backtest, StrategyInstance


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
