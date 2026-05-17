"""Trader-facing cockpit endpoints.

Each view is a thin wrapper around an `apps.portfolio.services.cockpits.*`
aggregation function. All return JSON-safe dicts. Reads only — no writes.

Routes are mounted **before** the PortfolioViewSet detail catchall in
``apps.portfolio.api.urls`` so paths like ``capital-cockpit/`` aren't
swallowed by ``^(?P<pk>[^/.]+)/$``.
"""
from __future__ import annotations

from datetime import date

from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.portfolio.services.cockpits import (
    build_broker_reconciliation,
    build_capital_cockpit,
    build_edge_decay,
    build_expiry_cockpit,
    build_greeks_heatmap,
    build_plan_vs_actual,
    build_regime_heatmap,
    build_risk_budget,
    build_signal_funnel,
    build_theta_forecast,
)
from apps.portfolio.services.correlation_matrix import build_correlation_report
from apps.portfolio.services.gap_risk import build_gap_risk_report
from apps.portfolio.services.post_mortem import build_post_mortem_report
from apps.portfolio.services.sizer_simulator import simulate as simulate_sizer


class _BaseCockpitView(APIView):
    permission_classes = [IsAuthenticated]


class CapitalCockpitView(_BaseCockpitView):
    """GET /api/v1/portfolios/capital-cockpit/"""
    def get(self, request):
        return Response(build_capital_cockpit(getattr(request, "tenant", None)))


class PlanVsActualView(_BaseCockpitView):
    """GET /api/v1/portfolios/plan-vs-actual/?limit=200&strategy=directional"""
    def get(self, request):
        limit = int(request.query_params.get("limit", 200) or 200)
        strategy = request.query_params.get("strategy") or None
        return Response(build_plan_vs_actual(
            getattr(request, "tenant", None), limit=limit, strategy=strategy,
        ))


class GreeksHeatmapView(_BaseCockpitView):
    """GET /api/v1/portfolios/greeks-heatmap/"""
    def get(self, request):
        return Response(build_greeks_heatmap(getattr(request, "tenant", None)))


class SignalFunnelView(_BaseCockpitView):
    """GET /api/v1/portfolios/signal-funnel/"""
    def get(self, request):
        return Response(build_signal_funnel(getattr(request, "tenant", None)))


class RiskBudgetView(_BaseCockpitView):
    """GET /api/v1/portfolios/risk-budget/?lookback_days=30"""
    def get(self, request):
        lookback = int(request.query_params.get("lookback_days", 30) or 30)
        return Response(build_risk_budget(
            getattr(request, "tenant", None), lookback_days=lookback,
        ))


class ExpiryCockpitView(_BaseCockpitView):
    """GET /api/v1/portfolios/expiry-cockpit/?underlying=NIFTY"""
    def get(self, request):
        underlying = (request.query_params.get("underlying") or "NIFTY").upper()
        return Response(build_expiry_cockpit(
            getattr(request, "tenant", None), underlying=underlying,
        ))


class BrokerReconView(_BaseCockpitView):
    """GET /api/v1/portfolios/broker-recon/?on_date=2026-05-14"""
    def get(self, request):
        raw = request.query_params.get("on_date")
        on_date = None
        if raw:
            try:
                on_date = date.fromisoformat(raw)
            except ValueError:
                on_date = None
        return Response(build_broker_reconciliation(
            getattr(request, "tenant", None), on_date=on_date,
        ))


class EdgeDecayView(_BaseCockpitView):
    """GET /api/v1/portfolios/edge-decay/?window=20"""
    def get(self, request):
        window = int(request.query_params.get("window", 20) or 20)
        return Response(build_edge_decay(
            getattr(request, "tenant", None), window=window,
        ))


class ThetaForecastView(_BaseCockpitView):
    """GET /api/v1/portfolios/theta-forecast/"""
    def get(self, request):
        return Response(build_theta_forecast(getattr(request, "tenant", None)))


class RegimeHeatmapView(_BaseCockpitView):
    """GET /api/v1/portfolios/regime-heatmap/"""
    def get(self, request):
        return Response(build_regime_heatmap(getattr(request, "tenant", None)))


class CorrelationMatrixView(_BaseCockpitView):
    """GET /api/v1/portfolios/correlation/"""
    def get(self, request):
        return Response(build_correlation_report(getattr(request, "tenant", None)))


class PostMortemView(_BaseCockpitView):
    """GET /api/v1/portfolios/post-mortem/?month=YYYY-MM"""
    def get(self, request):
        month = request.query_params.get("month") or None
        return Response(build_post_mortem_report(
            getattr(request, "tenant", None), month=month,
        ))


class GapRiskView(_BaseCockpitView):
    """GET /api/v1/portfolios/gap-risk/"""
    def get(self, request):
        return Response(build_gap_risk_report(getattr(request, "tenant", None)))


class SizerSimulatorView(_BaseCockpitView):
    """POST /api/v1/portfolios/sizer/simulate/
    body: {symbol, qty, side, stop, entry?, product?}
    """
    def post(self, request):
        return Response(simulate_sizer(request.data or {}))
