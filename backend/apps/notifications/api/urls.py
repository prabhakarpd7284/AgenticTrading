from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.notifications.api.tradingview_views import (
    GroupedSignalsView,
    TradingViewLinkViewSet,
    TradingViewWatchlistViewSet,
)
from apps.notifications.api.views import AlertViewSet

router = DefaultRouter()
# Alerts moved off the bare `/api/v1/notifications/` root onto `/alerts/` so
# the resource namespace can host sibling integrations (tradingview, future:
# slack/email destinations). Nothing in the FE was hitting the old root.
router.register("alerts", AlertViewSet, basename="alert")
router.register("tradingview", TradingViewLinkViewSet, basename="tradingview-link")
# Watchlists are scoped *under* /tradingview/watchlists/ rather than at
# /watchlists/ because they're tightly coupled to the TradingView feature
# — same operator surface, same Manager page. Easier to deprecate together
# than to split across two URL namespaces.
router.register(
    "tradingview/watchlists",
    TradingViewWatchlistViewSet,
    basename="tradingview-watchlist",
)

urlpatterns = [
    path("", include(router.urls)),
    # Aggregator endpoint for the TradingView Manager page. Standalone
    # because it spans multiple sources (not just TradingView) and benefits
    # from query-param-driven facets — fits neither viewset nicely.
    path(
        "tradingview/signals/",
        GroupedSignalsView.as_view(),
        name="tradingview-grouped-signals",
    ),
]
