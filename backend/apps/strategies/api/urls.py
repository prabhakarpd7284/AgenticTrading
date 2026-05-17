from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.strategies.api.views import StrategyInstanceViewSet, BacktestViewSet
from apps.trading.api import legacy_compat_views as _compat

router = DefaultRouter()
router.register("instances", StrategyInstanceViewSet, basename="strategy-instance")
router.register("backtests", BacktestViewSet, basename="backtest")

urlpatterns = [
    # /strategies/pyramid/backtest/ — runs the pyramid engine inline against
    # live broker data and returns a chart-ready payload (entries, exit, KPIs,
    # log). Was /legacy/pyramid/. Heavy synchronous call — for long iterations
    # use the Ops Console (`run_pyramid` via /ws/ops/).
    path("pyramid/backtest/", _compat.pyramid, name="pyramid-backtest"),
    path("", include(router.urls)),
]
