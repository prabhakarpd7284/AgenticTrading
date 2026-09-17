from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.trading.api.orders_views import OrderViewSet

router = DefaultRouter()
router.register("", OrderViewSet, basename="order")
urlpatterns = [path("", include(router.urls))]
