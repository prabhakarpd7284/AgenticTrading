"""Top-level URL conf. Each app owns its own /api/v1/<resource>/ namespace."""
from django.contrib import admin
from django.urls import include, path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

from apps.accounts.api.jwt import (
    TenantTokenObtainPairView,
    TenantTokenRefreshView,
)
from apps.notifications.api.tradingview_views import (
    TradingViewWebhookView, WatchlistKindsView, WatchlistViewSet,
)
from rest_framework.routers import DefaultRouter

_watchlists = DefaultRouter()
_watchlists.register("", WatchlistViewSet, basename="watchlist")

api_v1 = [
    # JWT views that embed the user's active tenant_id in the token claims —
    # without this, every downstream view fails the `TenantScoped` permission
    # and returns 403.  See apps/accounts/api/jwt.py for the rationale.
    path("auth/token/", TenantTokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("auth/token/refresh/", TenantTokenRefreshView.as_view(), name="token_refresh"),
    path("auth/", include("apps.accounts.api.urls")),
    path("tenants/", include("apps.tenants.api.urls")),
    path("billing/", include("apps.billing.api.urls")),
    # /brokers/ — broker linking was absorbed into market_data in Phase 4b.
    path("brokers/", include("apps.market_data.api.broker_urls")),
    path("market-data/", include("apps.market_data.api.urls")),
    # /portfolios/ + /orders/ — the portfolio, orders and trades apps were
    # merged into a single `trading` app in Phase 4c; their REST surfaces
    # stay split so frontend routes don't change.
    path("portfolios/", include("apps.trading.api.portfolio_urls")),
    path("orders/", include("apps.trading.api.orders_urls")),
    path("strategies/", include("apps.strategies.api.urls")),
    path("agents/", include("apps.agents_core.api.urls")),
    path("rag/", include("apps.rag.api.urls")),
    # /journals/ is a back-compat alias — Phase 4a folded the journals app
    # into events; the viewset now reads the unified Event log.
    path("journals/", include("apps.events.api.journal_urls")),
    path("events/", include("apps.events.api.urls")),
    path("notifications/", include("apps.notifications.api.urls")),
    # Watchlists live at top level because the primitive is domain-agnostic
    # (Setup page badges, autofire allowlists, screener universe, ...). The
    # model still lives in apps.notifications for historical reasons; only
    # the URL is hoisted.
    path("watchlists/kinds/", WatchlistKindsView.as_view(), name="watchlist-kinds"),
    path("watchlists/", include(_watchlists.urls)),
    # /ops/ — owner-only dev console: list management commands + stream
    # subprocess output over /ws/ops/ (see apps.system.consumers).
    path("ops/", include("apps.system.api.urls")),
    # /system/ — kill switch + AI pause/resume.
    path("system/", include("apps.system.api.system_urls")),
    # V2-native top-level endpoints (positions, trades, options-positions,
    # risk). The view functions still live in apps.trading.api.legacy_compat_views
    # for one more cycle while we refactor them into per-resource viewsets.
    path("", include("apps.trading.api.extras_urls")),
    # Public webhook receivers (no JWT — auth is the unguessable URL secret).
    # Kept under /api/v1/webhooks/ at the top so they don't inherit the
    # TenantScoped permission that every authenticated endpoint requires.
    path(
        "webhooks/tradingview/<str:secret>/",
        TradingViewWebhookView.as_view(),
        name="tradingview-webhook",
    ),
]

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include(api_v1)),
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
    path("healthz/", include("apps.common.health_urls")),
]
