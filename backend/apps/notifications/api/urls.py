from django.urls import include, path
from rest_framework.routers import DefaultRouter

from apps.notifications.api.views import AlertViewSet

router = DefaultRouter()
router.register("", AlertViewSet, basename="alert")
urlpatterns = [path("", include(router.urls))]
