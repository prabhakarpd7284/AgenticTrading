from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.strategies.api.views import (
    BacktestViewSet, BaseQualityView, BreakoutClassifierView,
    BTBootstrapView, BTCapacityView, BTCostSensitivityView, BTEdgeDriftView,
    BTMonteCarloView, BTRegimeStatsView, BTRegistryView, BTSweepView,
    BTWalkForwardView,
    MTFStageScannerView, StrategyInstanceViewSet,
)

router = DefaultRouter()
router.register("instances", StrategyInstanceViewSet, basename="strategy-instance")
router.register("backtests", BacktestViewSet, basename="backtest")
urlpatterns = [
    path("base-quality/", BaseQualityView.as_view(), name="base-quality"),
    path("mtf-stage/", MTFStageScannerView.as_view(), name="mtf-stage"),
    path("breakout-classifier/", BreakoutClassifierView.as_view(), name="breakout-classifier"),
    # Backtester Lab — 10 endpoints
    path("backtester/walk-forward/",      BTWalkForwardView.as_view(),    name="bt-walk-forward"),
    path("backtester/monte-carlo/",       BTMonteCarloView.as_view(),     name="bt-monte-carlo"),
    path("backtester/regime-stats/",      BTRegimeStatsView.as_view(),    name="bt-regime-stats"),
    path("backtester/cost-sensitivity/",  BTCostSensitivityView.as_view(),name="bt-cost-sensitivity"),
    path("backtester/edge-drift/",        BTEdgeDriftView.as_view(),      name="bt-edge-drift"),
    path("backtester/capacity/",          BTCapacityView.as_view(),       name="bt-capacity"),
    path("backtester/sweep/",             BTSweepView.as_view(),          name="bt-sweep"),
    path("backtester/bootstrap/",         BTBootstrapView.as_view(),      name="bt-bootstrap"),
    path("backtester/registry/",          BTRegistryView.as_view(),       name="bt-registry"),
    path("", include(router.urls)),
]
