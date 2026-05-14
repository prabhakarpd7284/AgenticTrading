"""Top-level URL conf. Each app owns its own /api/v1/<resource>/ namespace."""
from django.contrib import admin
from django.urls import include, path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

from apps.accounts.api.jwt import (
    TenantTokenObtainPairView,
    TenantTokenRefreshView,
)

api_v1 = [
    # JWT views that embed the user's active tenant_id in the token claims —
    # without this, every downstream view fails the `TenantScoped` permission
    # and returns 403.  See apps/accounts/api/jwt.py for the rationale.
    path("auth/token/", TenantTokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("auth/token/refresh/", TenantTokenRefreshView.as_view(), name="token_refresh"),
    path("auth/", include("apps.accounts.api.urls")),
    path("tenants/", include("apps.tenants.api.urls")),
    path("billing/", include("apps.billing.api.urls")),
    path("brokers/", include("apps.broker.api.urls")),
    path("market-data/", include("apps.market_data.api.urls")),
    path("portfolios/", include("apps.portfolio.api.urls")),
    path("orders/", include("apps.orders.api.urls")),
    path("strategies/", include("apps.strategies.api.urls")),
    path("agents/", include("apps.agents_core.api.urls")),
    path("rag/", include("apps.rag.api.urls")),
    # /journals/ is a back-compat alias — Phase 4a folded the journals app
    # into events; the viewset now reads the unified Event log.
    path("journals/", include("apps.events.api.journal_urls")),
    path("events/", include("apps.events.api.urls")),
    path("notifications/", include("apps.notifications.api.urls")),
    # Bridge to the legacy `trading` Django app — exposes the existing
    # 700+ rows of TradeJournal / StraddlePosition / AuditLog data so
    # the React UI shows real numbers immediately.  Will be folded into
    # the v2 endpoints once schema migration is complete.
    path("legacy/", include("apps.legacy.api.urls")),
]

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include(api_v1)),
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
    path("healthz/", include("apps.common.health_urls")),
]
