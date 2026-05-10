from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.strategies.api.views import StrategyInstanceViewSet, BacktestViewSet

router = DefaultRouter()
router.register("instances", StrategyInstanceViewSet, basename="strategy-instance")
router.register("backtests", BacktestViewSet, basename="backtest")
urlpatterns = [path("", include(router.urls))]
