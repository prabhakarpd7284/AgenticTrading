"""URL wiring for the portfolio app.

Same shadowing gotcha as ``apps.tenants.api.urls``: the ``""``-prefixed
PortfolioViewSet's detail route ``^(?P<pk>[^/.]+)/$`` would otherwise
swallow ``/positions/`` and ``/snapshots/``. Mounting sibling routers under
explicit path prefixes keeps them disjoint.
"""
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.portfolio.api.views import (
    MonthlyReportView,
    PortfolioViewSet,
    PositionViewSet,
    SnapshotViewSet,
)
from apps.portfolio.api.views_cockpits import (
    BrokerReconView,
    CapitalCockpitView,
    CorrelationMatrixView,
    EdgeDecayView,
    ExpiryCockpitView,
    ForcedFlatFlattenView,
    ForcedFlatView,
    GapRiskView,
    GreeksHeatmapView,
    PlanVsActualView,
    PostMortemView,
    RegimeHeatmapView,
    RiskBudgetView,
    SetCapitalView,
    SignalFunnelView,
    SizerSimulatorView,
    SlippageEdgeView,
    StructuralStopsView,
    ThetaForecastView,
)

portfolio_router = DefaultRouter()
portfolio_router.register("", PortfolioViewSet, basename="portfolio")

position_router = DefaultRouter()
position_router.register("", PositionViewSet, basename="position")

snapshot_router = DefaultRouter()
snapshot_router.register("", SnapshotViewSet, basename="snapshot")

urlpatterns = [
    # Specific paths first; catchall portfolio detail last.
    path("monthly/", MonthlyReportView.as_view(), name="portfolio-monthly"),
    # Cockpits — must precede the portfolio detail catchall router below.
    path("capital-cockpit/", CapitalCockpitView.as_view(), name="capital-cockpit"),
    path("capital/", SetCapitalView.as_view(), name="capital-set"),
    path("plan-vs-actual/", PlanVsActualView.as_view(), name="plan-vs-actual"),
    path("greeks-heatmap/", GreeksHeatmapView.as_view(), name="greeks-heatmap"),
    path("signal-funnel/", SignalFunnelView.as_view(), name="signal-funnel"),
    path("risk-budget/", RiskBudgetView.as_view(), name="risk-budget"),
    path("expiry-cockpit/", ExpiryCockpitView.as_view(), name="expiry-cockpit"),
    path("broker-recon/", BrokerReconView.as_view(), name="broker-recon"),
    path("edge-decay/", EdgeDecayView.as_view(), name="edge-decay"),
    path("theta-forecast/", ThetaForecastView.as_view(), name="theta-forecast"),
    path("regime-heatmap/", RegimeHeatmapView.as_view(), name="regime-heatmap"),
    path("correlation/", CorrelationMatrixView.as_view(), name="correlation-matrix"),
    path("post-mortem/", PostMortemView.as_view(), name="post-mortem"),
    path("gap-risk/", GapRiskView.as_view(), name="gap-risk"),
    path("sizer/simulate/", SizerSimulatorView.as_view(), name="sizer-simulate"),
    path("structural-stops/", StructuralStopsView.as_view(), name="structural-stops"),
    path("forced-flat/", ForcedFlatView.as_view(), name="forced-flat"),
    path("forced-flat/flatten/", ForcedFlatFlattenView.as_view(), name="forced-flat-flatten"),
    path("slippage-edge/", SlippageEdgeView.as_view(), name="slippage-edge"),
    path("positions/", include(position_router.urls)),
    path("snapshots/", include(snapshot_router.urls)),
    path("", include(portfolio_router.urls)),
]
