from rest_framework import serializers, viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.portfolio.models import Portfolio, PortfolioSnapshot, Position
from apps.portfolio.services.monthly_report import build_monthly_report


class PortfolioSerializer(serializers.ModelSerializer):
    class Meta:
        model = Portfolio
        fields = "__all__"


class PositionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Position
        fields = "__all__"


class SnapshotSerializer(serializers.ModelSerializer):
    class Meta:
        model = PortfolioSnapshot
        fields = "__all__"


class PortfolioViewSet(viewsets.ModelViewSet):
    serializer_class = PortfolioSerializer

    def get_queryset(self):
        return Portfolio.objects.filter(tenant=self.request.tenant)

    def perform_create(self, serializer):
        serializer.save(tenant=self.request.tenant)


class PositionViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = PositionSerializer

    def get_queryset(self):
        return Position.objects.filter(tenant=self.request.tenant)


class SnapshotViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = SnapshotSerializer

    def get_queryset(self):
        return PortfolioSnapshot.objects.filter(tenant=self.request.tenant)


class MonthlyReportView(APIView):
    """GET /api/v1/portfolios/monthly/

    Full monthly feedback report — trades taken, signals detected,
    capture rate per stock, risk rejections, and lessons learned.
    Cached 120s. Use ``?force=1`` to bypass.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        month = request.query_params.get("month")  # "2026-04" or None
        force = request.query_params.get("force") in ("1", "true")
        portfolio_id = request.query_params.get("portfolio")

        qs = Portfolio.objects.filter(tenant=request.tenant)
        if portfolio_id:
            qs = qs.filter(id=portfolio_id)
        portfolio = qs.first()

        if not portfolio:
            # Auto-create a default paper portfolio so the page always loads
            portfolio = Portfolio.objects.create(
                tenant=request.tenant,
                name="Default",
                capital=500_000,
                mode="paper",
            )

        payload = build_monthly_report(
            request.tenant, portfolio, month=month, force=force,
        )
        return Response(payload)
