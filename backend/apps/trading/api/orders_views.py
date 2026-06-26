from django.conf import settings
from rest_framework import serializers, status, viewsets
from rest_framework.response import Response

from apps.common.di import container
from apps.trading.domain.entities import OrderDraft
from apps.trading.models import Order, Portfolio
from apps.trading.services.place_order import PlaceOrder
from apps.trading.services.risk_engine import RiskEngine

# Wire default binding on import (can be overridden in tests).
container.bind(PlaceOrder, lambda: PlaceOrder(risk=RiskEngine()))


def _derive_idempotency_key(tenant_id, vd: dict) -> str:
    """Stable key for an identical order within a ~10s window — protects against
    an accidental double-submit (double-click / client retry) when the caller
    sends no Idempotency-Key header. An explicit header always wins."""
    import hashlib
    import time

    bucket = int(time.time()) // 10
    raw = (
        f"{tenant_id}:{vd['portfolio_id']}:{vd['symbol']}:{vd['side']}:"
        f"{vd['qty']}:{vd.get('price')}:{vd.get('order_type')}:{bucket}"
    )
    return "auto-" + hashlib.sha256(raw.encode()).hexdigest()[:40]


class OrderReadSerializer(serializers.ModelSerializer):
    class Meta:
        model = Order
        fields = "__all__"


class OrderCreateSerializer(serializers.Serializer):
    portfolio_id = serializers.UUIDField()
    symbol = serializers.CharField()
    side = serializers.ChoiceField(choices=["BUY", "SELL"])
    qty = serializers.IntegerField(min_value=1, max_value=settings.ALPHADESK["MAX_ORDER_QTY"])
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
        vd = ser.validated_data

        portfolio = Portfolio.objects.get(tenant=request.tenant, id=vd["portfolio_id"])
        # Explicit header wins; otherwise derive a short-window key so a
        # header-less double-submit still dedupes to one order. (#15)
        idem = request.headers.get("Idempotency-Key", "") or _derive_idempotency_key(
            request.tenant.id, vd,
        )
        draft = OrderDraft(**{k: v for k, v in vd.items() if k != "portfolio_id"})
        use_case = container.resolve(PlaceOrder)
        result = use_case.execute(
            tenant=request.tenant,
            user=request.user,
            portfolio=portfolio,
            draft=draft,
            idempotency_key=idem,
        )
        return Response(result.model_dump(), status=status.HTTP_202_ACCEPTED)
