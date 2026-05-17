from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.strategies.api.views import (
    BacktestViewSet, BaseQualityView, BreakoutClassifierView,
    MTFStageScannerView, StrategyInstanceViewSet,
)

router = DefaultRouter()
router.register("instances", StrategyInstanceViewSet, basename="strategy-instance")
router.register("backtests", BacktestViewSet, basename="backtest")
urlpatterns = [
    path("base-quality/", BaseQualityView.as_view(), name="base-quality"),
    path("mtf-stage/", MTFStageScannerView.as_view(), name="mtf-stage"),
    path("breakout-classifier/", BreakoutClassifierView.as_view(), name="breakout-classifier"),
    path("", include(router.urls)),
]
