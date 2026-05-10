"""Trimmed URL conf for the legacy Django process on :8001.

Exposes only what the sqlite file can answer for:
  * /api/v1/legacy/*     — read-only bridge to trading.* models
  * /api/v1/auth/*       — JWT login/refresh (so the SPA can still authenticate
                           against this process in isolation if needed)
  * /healthz/            — liveness probe
"""
from django.contrib import admin
from django.urls import include, path

from apps.accounts.api.jwt import (
    TenantTokenObtainPairView,
    TenantTokenRefreshView,
)

api_v1 = [
    path("auth/token/", TenantTokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("auth/token/refresh/", TenantTokenRefreshView.as_view(), name="token_refresh"),
    path("auth/", include("apps.accounts.api.urls")),
    path("legacy/", include("apps.legacy.api.urls")),
]

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include(api_v1)),
    path("healthz/", include("apps.common.health_urls")),
]
