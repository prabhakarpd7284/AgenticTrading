from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.strategies.api.views import (
    BacktestViewSet, BaseQualityView, StrategyInstanceViewSet,
)

router = DefaultRouter()
router.register("instances", StrategyInstanceViewSet, basename="strategy-instance")
router.register("backtests", BacktestViewSet, basename="backtest")
urlpatterns = [
    path("base-quality/", BaseQualityView.as_view(), name="base-quality"),
    path("", include(router.urls)),
]
