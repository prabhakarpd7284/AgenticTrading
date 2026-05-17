"""URL routing for the legacy bridge.

Mounted at /api/v1/legacy/ — see backend/config/urls.py.
"""
from django.urls import path

from . import views

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
    path("stock-summary/", views.stock_summary, name="stock-summary"),
    path("expiries/",      views.expiries,      name="expiries"),
]
