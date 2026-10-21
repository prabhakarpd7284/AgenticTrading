"""Portfolio destination API (UX redesign R2, plan §5).

  GET /api/v1/portfolio/overview/      whole-book picture, per tenant
  GET /api/v1/portfolio/attribution/   realized P&L by strategy / symbol / month (?months=12)

Both are tenant-scoped through ``request.tenant``; nothing here reads another
tenant's rows. Missing data comes back as ``None`` with a ``not_connected``
status, never as 0 (see ``apps.trading.services.portfolio_overview``).
"""
from __future__ import annotations

from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.common.permissions import TenantScoped
from apps.trading.services.portfolio_overview import build_attribution, build_overview


class PortfolioOverviewView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]

    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request):
        return Response(build_overview(request.tenant))


class PortfolioAttributionView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]

    @extend_schema(
        parameters=[OpenApiParameter("months", OpenApiTypes.INT, OpenApiParameter.QUERY, required=False,
                                     description="Window in months (1–60, default 12).")],
        responses=OpenApiTypes.OBJECT,
    )
    def get(self, request):
        try:
            months = min(max(int(request.query_params.get("months", 12)), 1), 60)
        except (TypeError, ValueError):
            months = 12
        return Response(build_attribution(request.tenant, months=months))
