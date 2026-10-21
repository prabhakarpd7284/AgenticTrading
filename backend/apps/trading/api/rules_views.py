"""/api/v1/trading-rules/ — rule state per portfolio (read-only).

  GET cooldown/   losing-streak cooldown status for each of the tenant's
                  trading portfolios (evaluated live from closed trades,
                  so it is never stale) + the rule constants.
"""
from __future__ import annotations

from django.utils import timezone
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.common.permissions import TenantScoped
from apps.trading.services.cooldown import evaluate_for_portfolio, rules_setting, trading_portfolios


class CooldownStatusView(APIView):
    permission_classes = [IsAuthenticated, TenantScoped]

    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request):
        today = timezone.localdate()
        out = []
        for p in trading_portfolios(request.tenant).order_by("created_at"):
            st = evaluate_for_portfolio(p, today)
            out.append({"portfolio_id": str(p.id), "portfolio": p.name, "mode": p.mode,
                        **st.as_dict()})
        return Response({
            "as_of": today.isoformat(),
            "results": out,
            "rules": {
                "loss_days": rules_setting("COOLDOWN_LOSS_DAYS"),
                "cooldown_trading_days": rules_setting("COOLDOWN_TRADING_DAYS"),
                "reduced_risk_multiplier": rules_setting("REDUCED_RISK_MULTIPLIER"),
                "costs_deducted": bool(rules_setting("DEDUCT_ESTIMATED_COSTS")),
            },
        })
