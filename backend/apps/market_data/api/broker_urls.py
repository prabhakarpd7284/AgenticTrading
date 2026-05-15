from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.market_data.api.broker_views import BrokerLinkViewSet

router = DefaultRouter()
router.register("", BrokerLinkViewSet, basename="broker-link")
urlpatterns = [path("", include(router.urls))]
