from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.strategies.api.views import StrategyInstanceViewSet, BacktestViewSet
from apps.strategies.api.scalp_views import create_scalp_run, scalp_defaults
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
    # /strategies/scalp/runs/ — create an interactive scalp sim run (no Celery);
    # client then opens ws/scalp/<run_id>/ to drive the playback.
    path("scalp/runs/", create_scalp_run, name="scalp-create-run"),
    path("scalp/defaults/", scalp_defaults, name="scalp-defaults"),
    path("", include(router.urls)),
]
