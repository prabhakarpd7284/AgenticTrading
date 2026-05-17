"""URL routing for the legacy bridge (deprecation phase).

The views moved to apps/trading/api/legacy_compat_views.py — this module
just re-routes the old /api/v1/legacy/* URLs at the same view functions
so the frontend keeps working during the migration to native v2 URLs
(see backend/config/urls.py for the new top-level routes).

The whole `apps.legacy` app is deleted in the final cutover commit
once the frontend no longer calls /legacy/* anywhere.
"""
from django.urls import path

from apps.trading.api import legacy_compat_views as views

app_name = "legacy"

urlpatterns = [
    path("portfolio/", views.portfolio, name="portfolio"),
    path("positions/", views.positions, name="positions"),
    path("trades/",    views.trades,    name="trades"),
    path("straddles/", views.straddles, name="straddles"),
    path("audit/",     views.audit,     name="audit"),
    path("risk/",      views.risk,      name="risk"),
    path("alerts/",    views.alerts,    name="alerts"),
    path("analytics/", views.analytics, name="analytics"),
    path("exposure/",  views.exposure,  name="exposure"),
    path("strategies/", views.strategies, name="strategies"),
    path("watchlist/", views.watchlist, name="watchlist"),
    path("system/",    views.system,    name="system"),
    path("ai/pause/",  views.pause_ai,  name="ai-pause"),
    path("ai/resume/", views.resume_ai, name="ai-resume"),
    path("pyramid/",   views.pyramid,   name="pyramid"),
]
