"""URL wiring for the portfolio app.

Same shadowing gotcha as ``apps.tenants.api.urls``: the ``""``-prefixed
PortfolioViewSet's detail route ``^(?P<pk>[^/.]+)/$`` would otherwise
swallow ``/positions/`` and ``/snapshots/``. Mounting sibling routers under
explicit path prefixes keeps them disjoint.
"""
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.trading.api.portfolio_views import (
    MonthlyReportView,
    PortfolioViewSet,
    PositionViewSet,
    SnapshotViewSet,
)
from apps.trading.api import legacy_compat_views as _compat

portfolio_router = DefaultRouter()
portfolio_router.register("", PortfolioViewSet, basename="portfolio")

position_router = DefaultRouter()
position_router.register("", PositionViewSet, basename="position")

snapshot_router = DefaultRouter()
snapshot_router.register("", SnapshotViewSet, basename="snapshot")

urlpatterns = [
    # Specific paths first; catchall portfolio detail last.
    path("monthly/", MonthlyReportView.as_view(), name="portfolio-monthly"),
    # Single-tenant aggregate (capital, used, day P&L, equity, snapshot).
    path("summary/", _compat.portfolio, name="portfolio-summary"),
    path("positions/", include(position_router.urls)),
    path("snapshots/", include(snapshot_router.urls)),
    path("", include(portfolio_router.urls)),
]
