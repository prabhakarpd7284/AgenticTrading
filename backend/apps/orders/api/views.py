from rest_framework import serializers, status, viewsets
from rest_framework.response import Response

from apps.common.di import container
from apps.orders.domain.entities import OrderDraft
from apps.orders.models import Order
from apps.orders.services.place_order import PlaceOrder
from apps.portfolio.models import Portfolio
from apps.trades.services.risk_engine import RiskEngine


# Wire default binding on import (can be overridden in tests).
container.bind(PlaceOrder, lambda: PlaceOrder(risk=RiskEngine()))


class OrderReadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Order
        fields = "__all__"


class OrderCreateSerializer(serializers.Serializer):
    portfolio_id = serializers.UUIDField()
    symbol = serializers.CharField()
    side = serializers.ChoiceField(choices=["BUY", "SELL"])
    qty = serializers.IntegerField(min_value=1)
    order_type = serializers.CharField(default="MARKET")
    product = serializers.CharField(default="INTRADAY")
    price = serializers.FloatField(required=False, allow_null=True)
    sl = serializers.FloatField(required=False, allow_null=True)
    tp = serializers.FloatField(required=False, allow_null=True)
    origin = serializers.CharField(default="ui")


class OrderViewSet(viewsets.ModelViewSet):
    serializer_class = OrderReadSerializer
    http_method_names = ["get", "post", "delete", "head", "options"]

    def get_queryset(self):
        return Order.objects.filter(tenant=self.request.tenant)

    def create(self, request, *args, **kwargs):
        ser = OrderCreateSerializer(data=request.data)
        ser.is_valid(raise_exception=True)

        portfolio = Portfolio.objects.get(
            tenant=request.tenant, id=ser.validated_data.pop("portfolio_id"),
        )
        draft = OrderDraft(**ser.validated_data)
        use_case = container.resolve(PlaceOrder)
        result = use_case.execute(
            tenant=request.tenant,
            user=request.user,
            portfolio=portfolio,
            draft=draft,
            idempotency_key=request.headers.get("Idempotency-Key", ""),
        )
        return Response(result.model_dump(), status=status.HTTP_202_ACCEPTED)
