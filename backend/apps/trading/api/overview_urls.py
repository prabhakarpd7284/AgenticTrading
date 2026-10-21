"""/api/v1/portfolio/ — the Portfolio destination (UX redesign R2)."""
from django.urls import path

from apps.trading.api.overview_views import PortfolioAttributionView, PortfolioOverviewView

urlpatterns = [
    path("overview/", PortfolioOverviewView.as_view(), name="portfolio-overview"),
    path("attribution/", PortfolioAttributionView.as_view(), name="portfolio-attribution"),
]
