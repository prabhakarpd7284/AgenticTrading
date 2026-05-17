"""V2-native URLs that bridge the legacy /api/v1/legacy/* surface area into
the proper apps.trading namespace. These read from v2 Postgres tables;
the underlying view functions live in `legacy_compat_views` and will be
refactored into proper ViewSets in a follow-up once the frontend has
migrated off the /legacy/ prefix.

Mounted at /api/v1/ via config/urls.py — see each `path()` for the full
URL.
"""
from django.urls import path

from apps.trading.api import legacy_compat_views as views

urlpatterns = [
    # /api/v1/positions/  — computed equity + options position overview.
    path("positions/",          views.positions, name="trading-positions"),
    # /api/v1/trades/      — Trade rows (was /legacy/trades/).
    path("trades/",             views.trades,    name="trading-trades"),
    # /api/v1/options-positions/  — OptionsPosition rows + nested legs.
    path("options-positions/",  views.straddles, name="trading-options-positions"),
    # /api/v1/risk/        — capital utilisation + daily-loss caps.
    path("risk/",               views.risk,      name="trading-risk"),
    # /api/v1/risk/alerts/ — derived alerts (loss caps, expiring options).
    path("risk/alerts/",        views.alerts,    name="trading-risk-alerts"),
]
