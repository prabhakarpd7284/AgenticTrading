from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.market_data.api.broker_views import BrokerLinkViewSet
from apps.market_data.api.combined_views import CombinedPositionsView
from apps.market_data.api.oauth_views import (
    oauth_callback, oauth_pending, oauth_reauth, oauth_start,
)

router = DefaultRouter()
router.register("", BrokerLinkViewSet, basename="broker-link")
urlpatterns = [
    # Multi-broker aggregated positions/holdings/margin.
    path("combined/positions/", CombinedPositionsView.as_view(), name="combined-positions"),
    # OAuth-redirect flow for Zerodha + Fyers.
    path("<str:broker_name>/oauth/start/",    oauth_start,    name="broker-oauth-start"),
    path("<str:broker_name>/oauth/callback/", oauth_callback, name="broker-oauth-callback"),
    path("<str:pk>/oauth/reauth/",            oauth_reauth,   name="broker-oauth-reauth"),
    path("oauth/pending/",                    oauth_pending,  name="broker-oauth-pending"),
    path("", include(router.urls)),
]
