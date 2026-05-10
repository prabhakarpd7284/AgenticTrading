from rest_framework.routers import DefaultRouter
from django.urls import include, path

from apps.accounts.api.views import AuthViewSet, MeViewSet

router = DefaultRouter()
router.register("auth", AuthViewSet, basename="auth")
router.register("me", MeViewSet, basename="me")

urlpatterns = [path("", include(router.urls))]
